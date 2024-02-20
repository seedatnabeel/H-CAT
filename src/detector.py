# third party
import numpy as np
import torch
import torch.nn.functional as F

from .models import LSTM


def predict_detector(train_dyn, epochs, device):
    """
    This function loads a pre-trained LSTM model and uses it to predict whether input training dynamics contains
    noise or not.

    The detector LSTM is trained based on https://github.com/Christophe-Jia/mislabel-detection

    Args:
      train_dyn: The input training dynamics for the detector model, which is a numpy array of shape (num_samples,
    sequence_length, input_dim).
      epochs: The number of epochs used to train the LSTM noise detector model.
      device: The device on which to run the detector model.

    Returns:
      a numpy array of binary predictions (0 or 1) based on whether the input data (train_dyn) is
    considered noisy or not by the LSTM noise detector model. The predictions are based on a threshold
    of 0.5 on the probability of the input being noisy, which is obtained by passing the input through
    the noise detector model.
    """

    print(f"Epochs {epochs}, Shape {train_dyn.shape}")

    detector_files = (
        f"src/detectors/{epochs}/cifar10.0.2_epochs{epochs}_lstm_detector.pth.tar"
    )
    noise_detector = LSTM(in_dim=train_dyn.shape[-1], hidden_dim=64, n_layer=2)
    noise_detector.load_state_dict(torch.load(detector_files)["state_dict"])

    noise_detector.eval()

    noise_detector.to(device)

    train_dyn = torch.tensor(train_dyn, device=device)

    out = noise_detector(train_dyn.float())
    out = F.softmax(out)
    out = out.cpu().detach().numpy()

    return np.array(out[:, 1] > 0.5).astype(int)
