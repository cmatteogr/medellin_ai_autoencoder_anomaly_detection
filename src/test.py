"""
Author: Cesar M. Gonzalez

Test Autoencoder anomaly detection model
"""

from src.autoencoder_model import Autoencoder
import torch
import torch.nn as nn


def test_autoencoder(model: Autoencoder, properties_data) -> dict:
    """
    Test Autoencoder anomaly detection model
    :param model: anomaly detection model
    :param properties_data: Test data file path
    :return: test report
    """
    print('# Start testing autoencoder anomaly detection model')
    # Transform data to tensors
    tensor_data = torch.tensor(properties_data, dtype=torch.float32)

    # Apply reconstruction
    print('Evaluate model on test data')
    model.eval()
    criterion = nn.MSELoss()
    reconstruction_data = model(tensor_data)
    reconstruction_error = criterion(reconstruction_data, tensor_data)

    print(f'Reconstruction error, best score: {reconstruction_error:.8f}')
    # Build train report
    print('Create test report dict')
    report_dict = {
        'reconstruction_error': float(reconstruction_error)
    }

    print('End testing autoencoder anomaly detection model')
    # Return model filepath
    return report_dict
