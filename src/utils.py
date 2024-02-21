# third party
import augly.image as imaugs
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor


def seed_everything(seed: int):
    """
    This function sets the random seed for various libraries in Python to ensure reproducibility of
    results.

    Args:
      seed (int): The seed parameter
    """
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def apply_augly(image):
    """
    The function applies a set of image augmentations using the AugLy library and returns the augmented
    image as a tensor. It is used for the ALLS HCM for the augmentation

    Args:
      image: The input image that needs to be augmented.

    Returns:
      an augmented tensor image. The image is being transformed using a list of augmentations and then
    converted to a tensor using PyTorch's `transforms.ToTensor()` function.
    """

    AUGMENTATIONS = [
        imaugs.HFlip(),
        imaugs.RandomBrightness(),
        imaugs.RandomNoise(),
        imaugs.RandomPixelization(min_ratio=0.1, max_ratio=0.3),
    ]

    TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])
    aug_tensor_image = TENSOR_TRANSFORMS(image)
    return aug_tensor_image


def kl_divergence(p, q):
    """
    The function calculates the Kullback-Leibler divergence between two probability distributions.

    Args:
      p: The variable `p` represents a probability distribution. It could be a tensor or a numpy array
    containing probabilities of different events.
      q: The parameter q is a probability distribution that we are comparing to another probability
    distribution p using the Kullback-Leibler (KL) divergence formula. KL divergence measures the
    difference between two probability distributions.

    Returns:
      The function `kl_divergence` returns the Kullback-Leibler divergence between two probability
    distributions `p` and `q`.
    """
    return (p * (p / q).log()).sum(dim=-1)


def get_groups(confidence, uncertainty, data_iq_xthresh, data_iq_ythresh):
    """
    The function `get_groups` categorizes data into easy, ambiguous, and hard groups based on
    confidence, uncertainty, and threshold values.

    Args:
      confidence: A numpy array containing the confidence scores for each data point.
      uncertainty: Uncertainty refers to the level of uncertainty or lack of confidence in a prediction.
    In this function, it is used as one of the criteria for grouping data points into different categories.
      data_iq_xthresh: It is a threshold value for the uncertainty of the data. Any data point with an
    uncertainty value less than or equal to this threshold is considered "easy" to classify.
      data_iq_ythresh: data_iq_ythresh is a threshold value used to determine the confidence threshold
    for separating easy and hard training examples. It is used to set the lower and upper bounds for the
    confidence threshold.

    Returns:
      three arrays: easy_train, ambig_train, and hard_train.
    """

    thresh = data_iq_ythresh
    conf_thresh_low = thresh
    conf_thresh_high = 1 - thresh
    conf_thresh = 0.5

    x_thresh = data_iq_xthresh

    hard_train = np.where((confidence <= conf_thresh_low) & (uncertainty <= x_thresh))[
        0
    ]
    easy_train = np.where((confidence >= conf_thresh_high) & (uncertainty <= x_thresh))[
        0
    ]

    hard_easy = np.concatenate((hard_train, easy_train))
    ambig_train = []
    for id in range(len(confidence)):
        if id not in hard_easy:
            ambig_train.append(id)

    ambig_train = np.array(ambig_train)
    return easy_train, ambig_train, hard_train
