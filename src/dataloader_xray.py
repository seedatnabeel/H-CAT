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
    def __init__(self, dataset, perturbation_method="uniform", p=0.1, rule_matrix=None):
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

        from collections import Counter

        labels = []
        for idx in range(len(self.dataset)):
            label = self.dataset[idx]["lab"]
            labels.append(label[0])

        self.targets = labels

        if perturbation_method in ["uniform", "asymmetric", "adjacent", "instance"]:
            self.perturbs = self._generate_mislabels()
        elif perturbation_method in [
            "id_covariate",
            "ood_covariate",
            "domain_shift",
            "zoom_shift",
            "crop_shift",
            "far_ood",
        ]:
            self.perturbs = self._generate_shifts()

    def get_flag_ids(self):
        return self.flag_ids

    def get_severity_ids(self):
        return self.severity_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        indices = self.indices[idx]
        target = data_point[1]
        perturbed_target = target

        # # not sure if needed since anyway we've updated the perturbed target
        # if self.perturbation_method in [
        #     "uniform",
        #     "asymmetric",
        #     "adjacent",
        #     "instance",
        # ]:
        #     if idx in self.perturbs:
        #         assert self.perturbs[idx] == perturbed_target
        #         perturbed_target = self.perturbs[idx]

        return data_point[0], target, perturbed_target, indices

    def _generate_shifts(self):
        """
        This function generates perturbed versions of a given dataset based on different perturbation
        methods. It updates self.dataset with the perturbed data.
        """
        import random

        shifts = []
        self.severity_ids = np.zeros(len(self.dataset))

        if (
            self.perturbation_method == "id_covariate"
            or self.perturbation_method == "ood_covariate"
        ):

            id_buckets = [0.01, 0.1, 0.25]
            ood_buckets = [0.5, 1, 2]
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            self.severity_ids = np.zeros(len(self.dataset))

            perturbed_data = []
            for idx in range(len(self.dataset)):
                data, label = self.dataset[idx]["img"], self.dataset[idx]["lab"]

                if idx in self.flag_ids:
                    # Get the original image
                    data, label = self.dataset[idx]["img"], self.dataset[idx]["lab"]
                    data = torch.from_numpy(data)
                    if self.perturbation_method == "id_covariate":
                        chosen_index = random.randint(0, len(id_buckets) - 1)
                        val = id_buckets[chosen_index]
                        std_dev = val
                        self.severity_ids[idx] = chosen_index + 1
                    elif self.perturbation_method == "ood_covariate":
                        chosen_index = random.randint(0, len(ood_buckets) - 1)
                        val = ood_buckets[chosen_index]
                        std_dev = val
                        self.severity_ids[idx] = chosen_index + 1
                    # Add Gaussian noise with a standard deviation of 10%/50% of the pixel intensity range
                    noise = torch.randn(data.shape) * (std_dev * torch.max(data))
                    noisy_data = data + noise
                    # Clip noisy image to ensure pixel values remain within [0, 1]
                    noisy_data = torch.clamp(noisy_data, 0, 1)
                    # Add the perturbed image and its label to the list
                    perturbed_data.append(noisy_data)

                else:
                    perturbed_data.append(torch.from_numpy(data))

            # Convert the list of tensors to a flat tensor and create a TensorDataset
            flat_data = torch.stack(perturbed_data)
            labels = self.targets
            labels = torch.tensor(labels)
            self.dataset = torch.utils.data.TensorDataset(flat_data, labels)

        if (
            self.perturbation_method == "zoom_shift"
            or self.perturbation_method == "crop_shift"
            or self.perturbation_method == "far_ood"
        ):

            zoom_shift_buckets = [2, 5, 10]
            crop_shift_buckets = [5, 10, 20]

            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            perturbed_data = []
            if self.perturbation_method == "far_ood":
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
                data, label = self.dataset[idx]["img"], self.dataset[idx]["lab"]
                from tqdm import tqdm

                if idx in self.flag_ids:
                    # Get the original image
                    data, label = self.dataset[idx]["img"], self.dataset[idx]["lab"]
                    data = torch.from_numpy(data)
                    if self.perturbation_method == "zoom_shift":
                        chosen_index = random.randint(0, len(zoom_shift_buckets) - 1)
                        val = zoom_shift_buckets[chosen_index]
                        noisy_data = zoom_in(data, zoom_factor=val)
                        self.severity_ids[idx] = chosen_index + 1
                    elif self.perturbation_method == "crop_shift":
                        chosen_index = random.randint(0, len(crop_shift_buckets) - 1)
                        val = crop_shift_buckets[chosen_index]
                        noisy_data = shift_image(data, shift_amount=val)
                        self.severity_ids[idx] = chosen_index + 1
                    elif self.perturbation_method == "far_ood":
                        if data.shape[0] == 1:
                            noisy_data = replace_with_cifar10(data, cifar10_dataset)
                        elif data.shape[0] == 3:
                            noisy_data = replace_with_mnist(data, mnist_dataset)

                    # Add the perturbed image and its label to the list
                    perturbed_data.append(noisy_data)

                else:
                    perturbed_data.append(torch.from_numpy(data))

                n += 1

            # Convert the list of tensors to a flat tensor and create a TensorDataset
            flat_data = torch.stack(perturbed_data)
            labels = self.targets
            labels = torch.tensor(labels)
            self.dataset = torch.utils.data.TensorDataset(flat_data, labels)

        if self.perturbation_method == "domain_shift":
            domain_shift_bucket = [0.25, 0.5, 1]
            # Change texture of image
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )
            perturbed_data = []
            for idx in range(len(self.dataset)):
                data, label = self.dataset[idx]["img"], self.dataset[idx]["lab"]
                data = torch.from_numpy(data)
                if idx in self.flag_ids:
                    chosen_index = random.randint(0, len(domain_shift_bucket) - 1)
                    val = domain_shift_bucket[chosen_index]
                    std_dev = val
                    self.severity_ids[idx] = chosen_index + 1
                    # COVARIATE SHIFT: Add Gaussian noise with a standard deviation of 10%/50% of the pixel intensity range

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
                    perturbed_data.append(data=torch.from_numpy(data))

            # Convert the list of tensors to a flat tensor and create a TensorDataset
            flat_data = torch.stack(perturbed_data)
            labels = self.targets
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
        print("pre = ", self.targets[0:10])

        if self.perturbation_method == "uniform":
            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )

            self.flag_ids = np.array(self.flag_ids, dtype=int)
            # self.targets = np.array(self.targets, dtype=int)

            new_labels = np.random.randint(
                0, len(np.unique(self.targets)), size=self.flag_ids.shape
            )
            try:
                self.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.targets).to(torch.long)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.targets = corrupt_labels.tolist()

        elif self.perturbation_method == "asymmetric":
            labels = np.array(self.targets)
            n_classes = len(np.unique(labels))
            self.flag_ids, new_labels = asymmetric_mislabeling(
                labels, self.p, n_classes
            )
            print(
                len(self.flag_ids) / len(self.dataset),
                len(new_labels) / len(self.dataset),
            )
            try:
                self.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.targets).to(torch.long)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.targets = corrupt_labels.tolist()

        elif self.perturbation_method == "instance":

            self.flag_ids = np.random.choice(
                len(self.dataset), int(len(self.dataset) * self.p), replace=False
            )

            y = np.array(self.targets)
            noisy_y = instance_mislabeling(
                y=y, flip_ids=self.flag_ids, rule_matrix=self.rule_matrix
            )
            new_labels = noisy_y[self.flag_ids]

            try:
                self.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.targets)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.targets = corrupt_labels.tolist()

        elif self.perturbation_method == "adjacent":
            labels = np.array(self.targets)
            n_classes = len(np.unique(labels))
            self.flag_ids, new_labels = adjacent_mislabeling(labels, self.p, n_classes)
            print(
                len(self.flag_ids) / len(self.dataset),
                len(new_labels) / len(self.dataset),
            )
            try:
                self.targets[self.flag_ids] = new_labels
            except:
                corrupt_labels = torch.tensor(self.targets)
                corrupt_labels[self.flag_ids] = torch.from_numpy(new_labels).to(
                    torch.long
                )
                self.targets = corrupt_labels.tolist()

        perturbed_data = []
        for idx in range(len(self.dataset)):
            data, label = self.dataset[idx]["img"], self.dataset[idx]["lab"]

            perturbed_data.append(torch.from_numpy(data))

        # Convert the list of tensors to a flat tensor and create a TensorDataset
        flat_data = torch.stack(perturbed_data)
        labels = self.targets
        labels = torch.tensor(labels)
        print("post = ", self.targets[0:10])
        self.dataset = torch.utils.data.TensorDataset(flat_data, labels)


class CustomDataset(Dataset):
    def __init__(self, data, target_column, transform=None, image_data=False):
        self.data = data
        self.target_column = target_column
        self.transform = transform
        self.image_data = image_data

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

        row = self.data.loc[idx, :].drop(self.target_column)
        target = self.data.loc[idx, self.target_column]

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
        data_type="csv",
        data_modality="image",
        batch_size=32,
        shuffle=True,
        num_workers=0,
        transform=None,
        image_transform=None,
        perturbation_method="uniform",
        p=0.1,
        rule_matrix=None,
    ):

        if data_type == "torch_dataset":
            self.dataset = data
        else:
            self.data = self._read_data(data, data_type)
            self.target_column = target_column

            if data_type == "image":
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

        self.rule_matrix = rule_matrix
        self.perturbed_dataset = PerturbedDataset(
            self.dataset,
            perturbation_method=perturbation_method,
            p=p,
            rule_matrix=self.rule_matrix,
        )
        self.flag_ids = self.perturbed_dataset.get_flag_ids()
        self.severity_ids = self.perturbed_dataset.get_severity_ids()
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

    def get_severity_ids(self):
        return self.severity_ids
