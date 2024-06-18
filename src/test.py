"""
Author: Cesar M. Gonzalez

Test Autoencoder anomaly detection model
"""

from src.autoencoder_model import Autoencoder
import torch


def test_autoencoder(model: Autoencoder, properties_data):
    """
    Test Autoencoder anomaly detection model
    :param model: anomaly detection model
    :param properties_data: Test data file path
    :return: test report
    """

    # Transform data to tensors
    tensor_data = torch.tensor(properties_data, dtype=torch.float32)

    # Apply reconstruction
    reconstruction_data = model(tensor_data)

    pass
