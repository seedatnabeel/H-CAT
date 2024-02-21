# stdlib
import os

# third party
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

from .perturbations import *


class PerturbedDataset(Dataset):
    def __init__(
        self,
        dataset,
        perturbation_method="uniform",
        p=0.1,
        rule_matrix=None,
        images=True,
        atypical_marginal=[],
    ):
        """
        This function initializes an object with a dataset, perturbation method, perturbation
        probability, and rule matrix, and generates mislabels or shifts based on the perturbation
        method.

        Args:
          dataset: The dataset parameter is the input dataset that the perturbation method will be
        applied to. It could be a numpy array or a pandas dataframe.
          perturbation_method: A string indicating the type of perturbation method to use. It can be one
        of the following: "uniform", "asymmetric", "adjacent", "instance", "id_covariate",
        "ood_covariate", "domain_shift", "zoom_shift", "crop_shift", or "far_. Defaults to uniform
          p: The probability of a label being perturbed/mislabelled.
          rule_matrix: The `rule_matrix` parameter is a matrix that represents the rules for perturbing
        the labels in the dataset. It is used in conjunction with the `perturbation_method` parameter to
        generate the mislabeled data. If `perturbation_method` is set to "rule_based", then `rule
        """
        self.dataset = dataset
        self.indices = np.array(range(len(dataset)))
        self.perturbation_method = perturbation_method
        self.p = p
        self.rule_matrix = rule_matrix
        self.images = images
        self.atypical_marginal = atypical_marginal

        if perturbation_method in ["uniform", "asymmetric", "adjacent", "instance"]:
            self.perturbs = self._generate_mislabels()
        elif perturbation_method in [
            "id_covariate",
            "ood_covariate",
            "domain_shift",
            "zoom_shift",
            "crop_shift",
            "far_ood",
            "atypical",
        ]:
            self.perturbs = self._generate_shifts()

    def get_flag_ids(self):
        return self.flag_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        indices = self.indices[idx]
        target = data_point[1]
        perturbed_target = target

        # not sure if needed since anyway we've updated the perturbed target
        if self.perturbation_method in [
            "uniform",
            "asymmetric",
            "adjacent",
            "instance",
        ]:
            if idx in self.perturbs:
                assert self.perturbs[idx] == perturbed_target
                perturbed_target = self.perturbs[idx]

        return data_point[0], target, perturbed_target, indices

    def _generate_shifts(self):
        """
        This function generates perturbed versions of a given dataset based on different perturbation
        methods. It updates self.dataset with the perturbed data.
        """

        shifts = []

        if (
            self.perturbation_method == "id_covariate"
            or self.perturbation_method == "ood_covariate"
        ):
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            perturbed_data = []
            for idx in range(len(self.dataset)):
                data, label = self.dataset[idx]
                if idx in self.flag_ids:
                    # Get the original image
                    data, label = self.dataset[idx]
                    if self.perturbation_method == "id_covariate":
                        std_dev = 0.1
                    elif self.perturbation_method == "ood_covariate":
                        std_dev = 0.5

                    if not self.images:
                        noise = np.random.normal(0, std_dev, size=data.shape)
                        # Add the noise to the data
                        noisy_data = data + noise
                    else:
                        # Add Gaussian noise with a standard deviation of 10%/50% of the pixel intensity range
                        noise = torch.randn(data.shape) * (std_dev * torch.max(data))
                        noisy_data = data + noise
                        # Clip noisy image to ensure pixel values remain within [0, 1]
                        noisy_data = torch.clamp(noisy_data, 0, 1)

                    # Add the perturbed image and its label to the list
                    perturbed_data.append(noisy_data)

                else:
                    perturbed_data.append(data)

            # Convert the list of tensors to a flat tensor and create a TensorDataset
            if not self.images:
                perturbed_data = np.array(perturbed_data)

                # Convert each to Tensor
                tensor_data = [torch.from_numpy(arr) for arr in perturbed_data]

                # Stack tuple of Tensors
                flat_data = torch.stack(tuple(tensor_data))
                # perturbed_data = torch.tensor(perturbed_data)
            else:
                # Convert the list of tensors to a flat tensor and create a TensorDataset
                flat_data = torch.stack(perturbed_data)
            labels = self.dataset.targets
            labels = torch.tensor(labels)
            self.dataset = torch.utils.data.TensorDataset(flat_data, labels)

        if (
            self.perturbation_method == "zoom_shift"
            or self.perturbation_method == "crop_shift"
            or self.perturbation_method == "far_ood"
            or self.perturbation_method == "atypical"
        ):
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            perturbed_data = []
            if self.perturbation_method == "far_ood":
                if self.images:
                    mnist_dataset = datasets.MNIST(
                        root="./data",
                        train=True,
                        download=True,
                        transform=transforms.ToTensor(),
                    )
                    cifar10_dataset = datasets.CIFAR10(
                        root="./data",
                        train=True,
                        download=True,
                        transform=transforms.ToTensor(),
                    )

            total = len(self.flag_ids)
            n = 0
            for idx in range(len(self.dataset)):
                data, label = self.dataset[idx]
                from tqdm import tqdm

                if idx in self.flag_ids:
                    # Get the original image
                    data, label = self.dataset[idx]
                    if self.perturbation_method == "zoom_shift":
                        noisy_data = zoom_in(data, zoom_factor=1.75)
                    elif self.perturbation_method == "crop_shift":
                        noisy_data = shift_image(data, shift_amount=10)
                    elif self.perturbation_method == "far_ood":
                        if self.images:
                            if data.shape[0] == 1:
                                noisy_data = replace_with_cifar10(data, cifar10_dataset)
                            elif data.shape[0] == 3:
                                noisy_data = replace_with_mnist(data, mnist_dataset)

                        else:
                            # Get indices of 0s and 1s
                            num_to_flip = 2
                            try:
                                zero_indices = np.where(data == 0)[0]
                                flip_zeros = np.random.choice(
                                    zero_indices, size=num_to_flip
                                )
                                data[flip_zeros] = 1
                            except:
                                pass

                            try:
                                one_indices = np.where(data == 1)[0]
                                flip_ones = np.random.choice(
                                    one_indices, size=num_to_flip
                                )
                                data[flip_ones] = 0
                            except:
                                pass

                            noisy_data = data
                    elif self.perturbation_method == "atypical":
                        if not self.images:
                            marginal, feat_idx = self.atypical_marginal

                            ns = int(0.01 * len(marginal))

                            # Sort the numpy array
                            sorted_arr = np.sort(marginal)

                            # Get the tail of the sorted array. For example, let's take the last 3 elements.
                            tail_elements = sorted_arr[-ns:]

                            # Sample from the tail. Here, sampling 2 elements without replacement.
                            sampled_val = np.random.choice(tail_elements, size=1)[0]
                            data[feat_idx] = sampled_val
                            noisy_data = data
                        else:
                            # Throw error that atypical is not supported for images --- need to use zoom_shift or crop_shift
                            raise ValueError(
                                "Atypical is not supported for images. Please use zoom_shift or crop_shift instead."
                            )

                    # Add the perturbed image and its label to the list
                    perturbed_data.append(noisy_data)

                else:
                    perturbed_data.append(data)

                n += 1

            # Convert the list of tensors to a flat tensor and create a TensorDataset
            if not self.images:
                perturbed_data = np.array(perturbed_data)

                # Convert each to Tensor
                tensor_data = [torch.from_numpy(arr) for arr in perturbed_data]

                # Stack tuple of Tensors
                flat_data = torch.stack(tuple(tensor_data))
                # perturbed_data = torch.tensor(perturbed_data)
            else:
                flat_data = torch.stack(perturbed_data)

            labels = self.dataset.targets
            labels = torch.tensor(labels)
            self.dataset = torch.utils.data.TensorDataset(flat_data, labels)

        if self.perturbation_method == "domain_shift":
            # Change texture of image
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            perturbed_data = []
            for idx in range(len(self.dataset)):
                data, label = self.dataset[idx]

                if idx in self.flag_ids:

                    # COVARIATE SHIFT: Add Gaussian noise with a standard deviation of 10%/50% of the pixel intensity range
                    std_dev = 0.5

                    noise = torch.randn(data.shape) * (std_dev * torch.max(data))
                    noisy_data = data + noise

                    # Clip noisy image to ensure pixel values remain within [0, 1]
                    noisy_data = torch.clamp(noisy_data, 0, 1)

                    # Domain shift: convert image to cartoon-like version
                    # Convert image to numpy array
                    data = np.array(to_pil_image(noisy_data))

                    if len(data.shape) == 3:  # check if image is 3-channel (RGB)

                        # Convert image to cartoon-like version using OpenCV
                        color_img = cv2.cvtColor(
                            data, cv2.COLOR_RGB2BGR
                        )  # convert to BGR for OpenCV
                        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                        gray_img = cv2.medianBlur(gray_img, 5)
                        edges = cv2.adaptiveThreshold(
                            gray_img,
                            255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY,
                            9,
                            9,
                        )
                        color_img = cv2.bilateralFilter(color_img, 9, 250, 250)
                        cartoon_img = cv2.bitwise_and(color_img, color_img, mask=edges)

                        # Convert cartoon-like image back to PyTorch tensor
                        cartoon_tensor_img = to_tensor(
                            cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB)
                        )
                    else:
                        # Image is already grayscale
                        gray_img = data
                        gray_img = cv2.medianBlur(gray_img, 5)
                        edges = cv2.adaptiveThreshold(
                            gray_img,
                            255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY,
                            9,
                            9,
                        )
                        color_img = cv2.bilateralFilter(gray_img, 9, 250, 250)
                        cartoon_img = cv2.bitwise_and(gray_img, gray_img, mask=edges)
                        cartoon_tensor_img = to_tensor(cartoon_img)

                    domain_shift_data = cartoon_tensor_img.unsqueeze(0)
                    if len(domain_shift_data.size()) == 4:
                        domain_shift_data = domain_shift_data.squeeze(0)
                    perturbed_data.append(domain_shift_data)

                else:
                    # Add the original image to the dataset
                    if len(data.size()) == 4:
                        data = data.squeeze(0)
                    perturbed_data.append(data)

            # Convert the list of tensors to a flat tensor and create a TensorDataset
            flat_data = torch.stack(perturbed_data)
            labels = self.dataset.targets
            labels = torch.tensor(labels)
            self.dataset = torch.utils.data.TensorDataset(flat_data, labels)

    def _generate_mislabels(self):
        """
        This function generates mislabeled data based on different perturbation methods.

        Returns:
          A dictionary containing the indices of mislabeled samples and their corresponding new labels
        is being returned.
        """
        mislabels = []

        if self.perturbation_method == "uniform":
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            new_labels = np.random.randint(
                0, len(np.unique(self.dataset.targets)), size=self.flag_ids.shape
            )
            try:
                self.dataset.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.dataset.targets)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.dataset.targets = corrupt_labels.tolist()

            return dict(zip(self.flag_ids, new_labels))

        elif self.perturbation_method == "asymmetric":
            labels = np.array(self.dataset.targets)
            n_classes = len(np.unique(labels))
            self.flag_ids, new_labels = asymmetric_mislabeling(
                labels, self.p, n_classes
            )
            print(
                len(self.flag_ids) / len(self.dataset),
                len(new_labels) / len(self.dataset),
            )
            try:
                self.dataset.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.dataset.targets)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.dataset.targets = corrupt_labels.tolist()

            return dict(zip(self.flag_ids, new_labels))

        elif self.perturbation_method == "instance":

            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )

            y = np.array(self.dataset.targets)
            noisy_y = instance_mislabeling(
                y=y, flip_ids=self.flag_ids, rule_matrix=self.rule_matrix
            )
            new_labels = noisy_y[self.flag_ids]

            try:
                self.dataset.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.dataset.targets)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.dataset.targets = corrupt_labels.tolist()

            return dict(zip(self.flag_ids, new_labels))

        elif self.perturbation_method == "adjacent":
            labels = np.array(self.dataset.targets)
            n_classes = len(np.unique(labels))
            self.flag_ids, new_labels = adjacent_mislabeling(labels, self.p, n_classes)
            print(
                len(self.flag_ids) / len(self.dataset),
                len(new_labels) / len(self.dataset),
            )
            try:
                self.dataset.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.dataset.targets)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.dataset.targets = corrupt_labels.tolist()

            return dict(zip(self.flag_ids, new_labels))


class CustomDataset(Dataset):
    def __init__(self, data, target_column, transform=None, image_data=False):
        self.data = data
        self.target_column = target_column
        self.transform = transform
        self.image_data = image_data

        if image_data == False:
            self.targets = data.iloc[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.image_data:
            image_path = self.data.loc[idx, self.target_column[0]]
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            target = self.data.loc[idx, self.target_column[1]]
            return image, target

        # row = self.data.loc[idx, :].drop(self.target_column)
        # target = self.data.loc[idx, self.target_column]

        row = self.data.iloc[idx, :-1].to_numpy()
        target = self.data.iloc[idx, -1]

        if self.transform:
            row, target = self.transform(row, target)

        return row, target


# The MultiFormatDataLoader class is a data loader that can handle multiple data formats, perturb the
# data, and shuffle the data for use in machine learning models.
class MultiFormatDataLoader:
    def __init__(
        self,
        data,
        target_column,
        data_type="torch_dataset",
        data_modality="image",
        batch_size=32,
        shuffle=True,
        num_workers=0,
        transform=None,
        image_transform=None,
        perturbation_method="uniform",
        p=0.1,
        rule_matrix=None,
        atypical_marginal=[],
    ):

        if data_type == "torch_dataset":
            self.dataset = data
        else:
            self.data = self._read_data(data, data_type)
            self.target_column = target_column

            # for not pytorch datasets
            if data_type == "raw_image":
                if image_transform is None:
                    image_transform = transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )

                self.dataset = CustomDataset(
                    self.data, target_column, transform=image_transform, image_data=True
                )
            else:
                self.dataset = CustomDataset(
                    self.data, target_column, transform=transform
                )

        if data_modality == "image":
            images = True
        else:
            images = False

        self.rule_matrix = rule_matrix
        self.perturbed_dataset = PerturbedDataset(
            self.dataset,
            perturbation_method=perturbation_method,
            p=p,
            rule_matrix=self.rule_matrix,
            images=images,
            atypical_marginal=atypical_marginal,
        )
        self.flag_ids = self.perturbed_dataset.get_flag_ids()
        self.dataloader = DataLoader(
            self.perturbed_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.dataloader_unshuffled = DataLoader(
            self.perturbed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def _read_data(self, data, data_type):
        """
        This function reads data in various formats

        Args:
          data: The data to be read, which could be in various formats depending on the data_type
        parameter.
          data_type: The type of data format being passed in, such as "csv", "json", "dict", "numpy",
        "tensor", or "image".

        Returns:
          a pandas DataFrame object that is created based on the input data and data type. The specific
        type of DataFrame returned depends on the data type specified in the input. If the data type is
        not supported, a ValueError is raised.
        """
        if data_type == "csv":
            return pd.read_csv(data)
        elif data_type == "json":
            return pd.read_json(data)
        elif data_type == "dict":
            return pd.DataFrame(data)
        elif data_type == "numpy":
            X, y = data
            return pd.DataFrame(np.column_stack((X, y)))
        elif data_type == "tensor":
            X, y = data
            return pd.DataFrame(torch.column_stack((X, y)).numpy())
        elif data_type == "image":
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {data_type}")

    def get_dataloader(self):
        """
        The function returns two dataloaders, one shuffled and one unshuffled.

        Returns:
          The method `get_dataloader` is returning a tuple containing two objects: `self.dataloader` and
        `self.dataloader_unshuffled`.
        """
        return self.dataloader, self.dataloader_unshuffled

    # get custom dataset
    def get_dataset(self):
        return self.dataset

    # get perturbed dataset
    def get_perturbed_dataset(self):
        return self.perturbed_dataset

    def get_flag_ids(self):
        """
        This function returns an array of zeros and ones indicating whether each index in a dataset is a
        flagged index or not.

        Returns:
          The function `get_flag_ids` returns a NumPy array of zeros with the same length as the perturbed
        dataset. The function then sets the value of 1 at the indices specified by the `flag_ids` attribute
        of the object. Finally, the function returns the updated flag array.
        """

        flag_array = np.zeros(len(self.perturbed_dataset))

        for i in range(len(flag_array)):
            if i in self.flag_ids:
                flag_array[i] = 1

        return flag_array
