# stdlib
import os
import random

# third party
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, to_tensor


def zoom_in(image, zoom_factor):
    """
    Zooms in on the image by the zoom factor

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        zoom_factor (float): Zoom factor (e.g., 1.5 for 1.5x zoom).

    Returns:
        torch.Tensor: Zoomed image tensor.
    """
    # Calculate the new size based on the zoom factor
    new_height = int(image.size(1) / zoom_factor)
    new_width = int(image.size(2) / zoom_factor)

    # Calculate the crop coordinates
    top = (image.size(1) - new_height) // 2
    left = (image.size(2) - new_width) // 2

    # Crop the region of interest
    cropped_image = F.crop(image, top, left, new_height, new_width)

    # Resize the cropped image to the original size using bilinear interpolation
    zoomed_image = F.resize(
        cropped_image,
        (image.size(1), image.size(2)),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )

    return zoomed_image


def shift_image(image, shift_amount=10):
    """
    Shifts the image right or left while maintaining the size.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        shift_amount (int): Number of pixels to shift the image.

    Returns:
        torch.Tensor: Shifted image tensor.
    """
    import random

    random_number = random.randint(0, 1)

    # Pad the image on the right or left side depending on the shift amount
    if shift_amount > 0:
        padded_image = F.pad(image, (shift_amount, 0))
    elif shift_amount < 0:
        padded_image = F.pad(image, (0, -shift_amount))
    else:
        return image

    # Crop the padded image to the original size
    shifted_image = padded_image[:, :, : image.size(2)]

    return shifted_image


def replace_with_cifar10(data, cifar10_dataset):
    """
    The function replaces an input image with a randomly selected grayscale image from the CIFAR-10
    dataset, resized to match the size of the input image.

    Args:
      data (torch.Tensor): The `data` parameter is a tensor representing an image from the MNIST dataset.
      cifar10_dataset: The `cifar_dataset` parameter is a dataset object containing the CIFAR-10 dataset. It
    is used to randomly select an image from the CIFAR dataset and replace a portion of the input data
    with that image.
    Returns:
      a tensor that represents a grayscale CIFAR-10 image that has been resized to match the size of the
    input tensor `data`.
    """
    # Load CIFAR-10 dataset

    # Randomly select an image from CIFAR-10
    cifar10_image, _ = cifar10_dataset[
        int(torch.randint(0, len(cifar10_dataset), (1,)))
    ]

    # Convert CIFAR-10 image to grayscale
    cifar10_image = cifar10_image.mean(dim=0, keepdim=True)

    # Resize CIFAR-10 image to match the size of MNIST image tensor
    resize_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((data.size(1), data.size(2))),
            transforms.ToTensor(),
        ]
    )
    cifar10_image = resize_transform(cifar10_image)

    return cifar10_image


def replace_with_mnist(data, mnist_dataset):
    """
    The function replaces a given image with a randomly selected image from the MNIST dataset, resized
    to match the size of the original image.

    Args:
      data (torch.Tensor): The `data` parameter is a tensor representing an image from the CIFAR-10 dataset.
      mnist_dataset (torch.Dataset): The `mnist_dataset` parameter is a dataset object containing the MNIST dataset. It
    is used to randomly select an image from the MNIST dataset and replace a portion of the input data
    with that image.

    Returns:
      a tensor representing an MNIST image that has been resized to match the size of a CIFAR-10 image
    tensor.
    """
    # Load MNIST dataset

    # Randomly select an image from MNIST
    mnist_image, _ = mnist_dataset[int(torch.randint(0, len(mnist_dataset), (1,)))]

    mnist_image = mnist_image.expand(3, -1, -1)

    # Resize MNIST image to match the size of CIFAR-10 image tensor
    resize_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((data.size(1), data.size(2))),
            transforms.ToTensor(),
        ]
    )
    mnist_image = resize_transform(mnist_image)

    return mnist_image


def instance_mislabeling(y, flip_ids, rule_matrix):
    """
    The function takes in a set of labels, indices to flip, and a rule matrix, and returns an instance flipped
    set of labels based on the rule matrix.

    Args:
      y: The original labels of the instances.
      flip_ids: `flip_ids` is a list of indices representing the instances in the dataset that need to
    be flipped. In other words, these are the instances whose labels need to be changed according to the
    `rule_matrix`.
      rule_matrix: The rule_matrix is a matrix that defines the possible label flips for each original
    label. It is defined by prior knowledge from the user

    Returns:
      The function `instance_noise_flipper` returns a numpy array `flipped_y` which is a copy of the
    input array `y` with some of its elements flipped according to the `flip_ids` and `rule_matrix`
    parameters.
    """
    flipped_indices = flip_ids
    flipped_y = np.copy(y)

    for idx in flipped_indices:
        label = y[idx]
        flipped_label = rule_matrix[label]

        flipped_y[idx] = np.random.choice(flipped_label)

    return flipped_y


def asymmetric_mislabeling(labels, p, n_classes):
    """
    The function generates asymmetric noise matrices for each class and adds noise to randomly selected
    labels based on the noise matrices.

    Args:
      labels: The original labels of the dataset, represented as a 1D numpy array.
      p: p is the proportion of labels to flip, i.e., the noise proportion. It should be a value between
    0 and 1.
      n_classes: The number of classes in the classification problem.

    Returns:
      two arrays: `flip_indices` and `flipped_labels`. `flip_indices` contains the indices of the labels
    that were flipped, and `flipped_labels` contains the new labels that were generated after applying
    the asymmetric noise.
    """
    if not 0 <= p <= 1:
        raise ValueError("Noise proportion p must be in the range [0, 1]")

    num_labels = len(labels)
    num_flips = int(round(p * num_labels))

    # Generate asymmetric noise matrices for each class
    noise_matrices = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        off_diag_elems = np.random.dirichlet(alpha=np.ones(n_classes - 1), size=1)
        off_diag_elems = np.insert(off_diag_elems, i, 0, axis=1)
        noise_matrices[i] = off_diag_elems
    np.fill_diagonal(noise_matrices, 0)
    noise_matrices /= noise_matrices.sum(axis=1, keepdims=True)

    # Select random indices to flip
    flip_indices = np.random.choice(num_labels, size=num_flips, replace=False)

    # Add noise to the selected labels
    original_labels = np.array(labels)
    flipped_labels = np.zeros(num_flips, dtype=int)
    for i, idx in enumerate(flip_indices):
        original_label = original_labels[idx]

        try:
            flipped_label = np.random.choice(
                n_classes, 1, p=noise_matrices[original_label]
            )
        except:
            flipped_label = np.random.choice(
                n_classes, 1, p=noise_matrices[int(original_label)]
            )
        flipped_labels[i] = flipped_label

    return np.array(flip_indices), np.array(flipped_labels)


def adjacent_mislabeling(labels, p, n_classes):
    """
    This function generates adjacent class flips in a given set of labels with a specified noise
    proportion.

    Args:
      labels: A list or array of integer labels representing the true class labels of a dataset.
      p: p is the proportion of labels to be flipped. It should be a value between 0 and 1.
      n_classes: The number of classes in the classification problem.

    Returns:
      two numpy arrays: flip_indices and flipped_labels. flip_indices contains the indices of the
    randomly selected labels that were flipped, and flipped_labels contains the new labels after
    applying the noise.
    """
    if not 0 <= p <= 1:
        raise ValueError("Noise proportion p must be in the range [0, 1]")

    num_labels = len(labels)
    num_flips = int(round(p * num_labels))

    # create adjacent noise matrix
    noise_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        noise_matrix[i][(i - 1) % n_classes] = 0.5
        noise_matrix[i][(i + 1) % n_classes] = 0.5

    # Select random indices to flip
    flip_indices = np.random.choice(num_labels, size=num_flips, replace=False)

    # Add noise to the selected labels
    original_labels = np.array(labels)
    flipped_labels = np.zeros(num_flips, dtype=int)
    for i, idx in enumerate(flip_indices):
        original_label = original_labels[idx]
        flipped_label = np.random.choice(n_classes, 1, p=noise_matrix[original_label])
        flipped_labels[i] = flipped_label

    return np.array(flip_indices), np.array(flipped_labels)
