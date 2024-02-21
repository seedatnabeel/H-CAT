# stdlib
import os

# third party
import augly.image as imaugs
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor

from .data_uncertainty import *
from .detector import predict_detector
from .utils import *


# Base class for HCMs
class Hardness_Base:
    def __init__(self, name):
        self.name = name
        self._scores = None

    def updates(self):
        pass

    def compute_scores(self):
        pass

    @property
    def scores(self):
        if self._scores is None:
            self.compute_scores()
        return self._scores


# This is a class that computes scores for Data-IQ and Data Maps
class DataIQ_Maps_Class(Hardness_Base):
    # Based on: https://github.com/seedatnabeel/Data-IQ
    # Data-IQ: https://arxiv.org/abs/2210.13043
    # Data Maps: https://arxiv.org/abs/2009.10795
    def __init__(self, dataloader, name="data_iq_maps"):
        super().__init__(name)
        self.data_eval = DataIQ_MAPS_Torch(dataloader=dataloader, sparse_labels=True)

    def updates(self, net, device):
        self.data_eval.on_epoch_end(net=net, device=device, gradient=False)

    def compute_scores(self, datamaps=True):
        if datamaps == True:
            self._scores = (self.data_eval.variability, self.data_eval.confidence)
        else:
            self._scores = (self.data_eval.aleatoric, self.data_eval.confidence)

        return self._scores


# This is a class that computes scores for detector.
class Detector_Class(Hardness_Base):
    # Based on: https://github.com/Christophe-Jia/mislabel-detection
    # https://arxiv.org/abs/2212.09321
    def __init__(self, name="detector"):
        super().__init__(name)

    def updates(self, data_uncert_class, device):
        self.data_uncert_class = data_uncert_class
        self.device = device

    def compute_scores(self):
        train_dyn = self.data_uncert_class.gold_labels_probabilities
        train_dyn = np.expand_dims(train_dyn, axis=2)
        self._scores = predict_detector(
            train_dyn=train_dyn, epochs=train_dyn.shape[1], device=self.device
        )

        return self._scores


# This is a class that computes scores for Cleanlab.
class Cleanlab_Class(Hardness_Base):
    # Based on: https://github.com/cleanlab/cleanlab
    # https://arxiv.org/abs/1911.00068
    def __init__(self, dataloader, name="cleanlab"):
        super().__init__(name)
        self.dataloader = dataloader
        self.logits = None
        self.targets = None
        self.probs = None

    def updates(self, logits, targets, probs):
        self.logits = logits
        self.targets = targets.detach().cpu().numpy()
        self.probs = probs.detach().cpu().numpy()

    def compute_scores(self):
        from cleanlab.count import num_label_issues
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores, order_label_issues

        scores = get_label_quality_scores(
            labels=self.targets,
            pred_probs=self.probs,
        )

        self._scores = scores

        self.num_errors = num_label_issues(
            labels=self.targets,
            pred_probs=self.probs,
        )

        return self._scores, self.num_errors


# This is a class that computes scores for EL2N.
class EL2N_Class(Hardness_Base):
    # Based on: https://github.com/BlackHC/pytorch_datadiet
    # Original jax: https://github.com/mansheej/data_diet
    # https://arxiv.org/abs/2107.07075
    def __init__(self, dataloader, name="el2n"):
        super().__init__(name)
        self.dataloader = dataloader
        self.unnormalized_model_outputs = None
        self.targets = None

    def updates(self, logits, targets):
        self.unnormalized_model_outputs = logits
        self.targets = targets
        # self.softmax_outputs = probs

    def compute_scores(self):
        import torch.nn.functional as F

        if len(self.targets.shape) == 1:
            self.targets = F.one_hot(
                self.targets, num_classes=self.unnormalized_model_outputs.size(1)
            ).float()

        # compute the softmax of the unnormalized model outputs
        softmax_outputs = F.softmax(self.unnormalized_model_outputs, dim=1)
        # compute the squared L2 norm of the difference between the softmax outputs and the target labels
        el2n_score = torch.sum((softmax_outputs - self.targets) ** 2, dim=1)
        self._scores = el2n_score.detach().cpu().numpy()


# This is a class that computes scores for AUM.
class AUM_Class(Hardness_Base):
    # Based on https://github.com/asappresearch/aum
    # https://arxiv.org/abs/2001.10528
    def __init__(self, name="aum", save_dir="."):
        super().__init__(name)
        from aum import AUMCalculator

        self.aum_calculator = AUMCalculator(save_dir, compressed=True)
        self.aum_scores = None

    def updates(self, y_pred, y_batch, sample_ids):
        # override method1
        records = self.aum_calculator.update(
            y_pred, y_batch.type(torch.int64), sample_ids.cpu().numpy()
        )

    def compute_scores(self):
        from os.path import exists

        if exists("aum_values.csv"):
            os.remove("aum_values.csv")

        self.aum_calculator.finalize()
        aum_df = pd.read_csv("aum_values.csv")
        self.aum_scores = []
        for i in range(aum_df.shape[0]):
            aum_sc = aum_df[aum_df["sample_id"] == i].aum.values[0]
            self.aum_scores.append(aum_sc)

        os.remove("aum_values.csv")
        self._scores = self.aum_scores

    def get_scores(self):
        return self._scores


# This is a class that computes scores for Forgetting scores.
class Forgetting_Class(Hardness_Base):
    # Based on: https://arxiv.org/abs/1812.05159
    def __init__(self, dataloader, total_samples, name="forgetting"):
        super().__init__(name)
        self.dataloader = dataloader
        self.total_samples = total_samples
        self.forgetting_counts = {i: 0 for i in range(self.total_samples)}
        self.last_remembered = {i: False for i in range(self.total_samples)}
        self.num_epochs = 0

    def updates(self, logits, targets, probs, indices):
        import torch.nn.functional as F

        softmax_outputs = F.softmax(logits, dim=1)
        _, predicted = torch.max(softmax_outputs.data, 1)
        predicted = predicted.detach().cpu().numpy()

        labels = targets.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        # Calculate forgetting events for the current batch
        for idx, (correct_prediction, index) in enumerate(
            zip(predicted == labels, indices)
        ):
            if correct_prediction and not self.last_remembered[index]:
                self.last_remembered[index] = True
            elif not correct_prediction and self.last_remembered[index]:
                self.forgetting_counts[index] += 1
                self.last_remembered[index] = False

        self.num_epochs += 1

    def compute_scores(self):
        total_forgetting_scores = np.zeros(self.total_samples)
        for idx in range(self.total_samples):
            total_forgetting_scores[idx] = self.forgetting_counts[idx] / (
                self.num_epochs
            )

        self._scores = total_forgetting_scores


# This is a class that computes scores for GraNd
class GRAND_Class(Hardness_Base):
    # Based on: https://github.com/BlackHC/pytorch_datadiet
    # Original jax: https://github.com/mansheej/data_diet
    # https://arxiv.org/abs/2107.07075
    def __init__(self, dataloader, name="grand"):
        super().__init__(name)
        self.dataloader = dataloader
        self.net = None
        self.device = None

    def updates(self, net, device):
        self.net = net
        self.device = device

    def compute_scores(self):
        import torch.nn.functional as F
        from functorch import grad, make_functional_with_buffers, vmap

        fmodel, params, buffers = make_functional_with_buffers(self.net)

        fmodel.eval()

        def compute_loss_stateless_model(params, buffers, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0).long()

            predictions = fmodel(params, buffers, batch)
            loss = F.cross_entropy(predictions, targets)
            return loss

        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

        print("Evaluating GRAND scores...")
        grad_norms = []

        for data in self.dataloader:
            inputs, _, targets, _ = data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            ft_per_sample_grads = ft_compute_sample_grad(
                params, buffers, inputs, targets
            )

            squared_norm = 0
            for param_grad in ft_per_sample_grads:
                squared_norm += param_grad.flatten(1).square().sum(dim=-1)
            grad_norms.append(squared_norm.detach().cpu().numpy() ** 0.5)

        grand_scores = np.concatenate(grad_norms, axis=0)
        self._scores = grand_scores


# This is a class that computes scores for VOG
class VOG_Class(Hardness_Base):
    # Based on https://github.com/chirag126/VOG
    # https://arxiv.org/abs/2008.11600
    def __init__(self, dataloader, total_samples, name="vog"):
        super().__init__(name)
        self.dataloader = dataloader
        self.total_samples = total_samples
        self.vog = {}

    def updates(self, net, device):

        net.eval()
        idx = 0
        for x, _, y, _ in self.dataloader:
            x = x.to(device)
            y = y.to(device)

            x.requires_grad = True
            sel_nodes_shape = y.shape
            ones = torch.ones(sel_nodes_shape).to(device)

            logits = net(x)
            probs = torch.nn.Softmax(dim=1)(logits)

            sel_nodes = probs[torch.arange(len(y)), y.type(torch.LongTensor)]
            sel_nodes.backward(ones)
            grad = x.grad.data.detach().cpu().numpy()

            for i in range(x.shape[0]):

                if idx not in self.vog.keys():
                    self.vog[idx] = []
                    self.vog[idx].append(grad[i, :].tolist())
                else:
                    self.vog[idx].append(grad[i, :].tolist())

                idx += 1

    def compute_scores(self, mode="complete"):
        # Analysis of the gradients
        training_vog_stats = []
        for i in range(len(self.vog)):
            if mode == "early":
                temp_grad = np.array(self.vog[i][:5])
            elif mode == "middle":
                temp_grad = np.array(self.vog[i][5:10])
            elif mode == "late":
                temp_grad = np.array(self.vog[i][10:])
            elif mode == "complete":
                temp_grad = np.array(self.vog[i])
            mean_grad = np.sum(np.array(self.vog[i]), axis=0) / len(temp_grad)
            training_vog_stats.append(
                np.mean(
                    np.sqrt(
                        sum([(mm - mean_grad) ** 2 for mm in temp_grad])
                        / len(temp_grad)
                    )
                )
            )

        self._scores = training_vog_stats


# This is a class that computes scores for Prototypicality
class Prototypicality_Class(Hardness_Base):
    # Based on: https://arxiv.org/abs/2206.14486
    def __init__(self, dataloader, num_classes, name="prototypicality"):
        super().__init__(name)
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.cosine_scores = []

    def updates(self, net, device):
        from collections import defaultdict

        import torch.nn.functional as F

        # Initialize accumulators for embeddings and counts for each label
        embeddings_dict = {i: [] for i in range(self.num_classes)}
        print("computing mean embeddings...")
        for batch_idx, data in enumerate(self.dataloader):
            x, _, y, _ = data
            x = x.to(device)
            y = y.to(device)
            embedding = net(x, embed=True)
            batch_size = x.size(0)

            for i in range(batch_size):
                label = int(y[i].detach().cpu().numpy())
                embeddings_dict[label].append(embedding[i])

        # Calculate the mean embeddings for each label
        mean_embeddings = {
            i: torch.stack(embeddings).mean(dim=0)
            for i, embeddings in embeddings_dict.items()
        }

        # Compute the cosine distance between each item in the dataloader and each key's mean in embeddings_sum
        print("Computing Cosine Distances...")
        for batch_idx, data in enumerate(self.dataloader):
            x, _, y, _ = data
            x = x.to(device)
            y = y.to(device)

            batch_size = x.size(0)

            for i in range(batch_size):
                label = int(y[i].detach().cpu().numpy())
                mean_embedding = mean_embeddings[label]
                cosine_similarity = F.cosine_similarity(
                    net(x[i : i + 1], embed=True), mean_embedding.unsqueeze(0), dim=1
                )
                self.cosine_scores.append(cosine_similarity.detach().cpu().item())

    def compute_scores(self):
        self._scores = self.cosine_scores


# This is a class that computes scores for ALLSH
class AllSH_Class(Hardness_Base):
    # Based on: https://arxiv.org/abs/2205.04980
    def __init__(self, dataloader, name="all_sh"):
        super().__init__(name)
        self.dataloader = dataloader
        self.kl_divergences = []

    def updates(self, net, device):
        net.eval()

        to_pil = ToPILImage()

        with torch.no_grad():
            for data in self.dataloader:
                images, _, _, _ = data
                images = images.to(device)

                # Compute the softmax for the original images

                logits = net(images)
                softmax_original = F.softmax(logits, dim=1)

                # Apply AugLy augmentations and compute the softmax for the augmented images
                augmented_images = []
                for image in images:
                    pil_image = to_pil(image.cpu())
                    augmented_image = apply_augly(pil_image)
                    augmented_images.append(augmented_image)
                augmented_images = torch.stack(augmented_images).to(device)

                logits_aug = net(augmented_images)
                softmax_augmented = F.softmax(logits_aug, dim=1)

                # Compute the KL divergence between the softmaxes and store it in the list
                kl_div = kl_divergence(softmax_original, softmax_augmented)
                self.kl_divergences.extend(kl_div.cpu().numpy().tolist())

    def compute_scores(self):
        self._scores = self.kl_divergences
        return self._scores


# This is a class that computes scores for Large Loss
class Large_Loss_Class(Hardness_Base):
    # Based on: https://arxiv.org/abs/2106.00445
    def __init__(self, dataloader, name="large_loss"):
        super().__init__(name)
        self.dataloader = dataloader
        self.losses = []

    def updates(self, logits, targets):
        # compute the loss for each sample separately
        epoch_losses = []
        for i in range(len(logits)):
            loss = F.cross_entropy(logits[i].unsqueeze(0), targets[i].unsqueeze(0))
            epoch_losses.append(loss.detach().cpu().item())

        self.losses.append(epoch_losses)

    def compute_scores(self):
        self._scores = np.mean(self.losses, axis=0)
        return self._scores


# This is a class that computes scores for Confidence Agreement
class Conf_Agree_Class(Hardness_Base):
    # Based on: https://arxiv.org/abs/1910.13427
    def __init__(self, dataloader, name="conf_agree"):
        super().__init__(name)
        self.dataloader = dataloader
        self.mean_scores = []

    def updates(self, net, device):
        net.train()

        with torch.no_grad():
            for data in self.dataloader:
                images, _, _, _ = data
                images = images.to(device)
                mc_softmax_outputs = []

                # Perform Monte Carlo Dropout 10 times
                for _ in range(10):
                    logits = net(images)
                    softmax_output = F.softmax(logits, dim=1)
                    mc_softmax_outputs.append(softmax_output)

                # Stack the softmax outputs and compute the mean along the Monte Carlo samples dimension
                mc_softmax_outputs = torch.stack(mc_softmax_outputs, dim=0)
                mean_softmax_output = torch.mean(mc_softmax_outputs, dim=0)

                # Compute and store the mean confidence for each sample in the dataset
                max_values, _ = torch.max(mean_softmax_output, dim=1)
                self.mean_scores.extend(max_values.cpu().numpy().tolist())

    def compute_scores(self):
        self._scores = self.mean_scores
        return self._scores
