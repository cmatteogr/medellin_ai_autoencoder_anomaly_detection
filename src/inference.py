"""
Author: Cesar M. Gonzalez

Train Autoencoder anomaly detection model
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(Model):
    """
    Parameters
    ----------
    output_units: int
      Number of output units

    code_size: int
      Number of units in bottle neck
    """

    def __init__(self, output_units, code_size=8):
        super().__init__()
        self.encoder = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(code_size, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def train(data_filepath: str, scaler_filepath: str) -> None:
    """
    Train Autoencoder anomaly detection model
    :param data_filepath: Data file path
    :param scaler_filepath: MinMax scaler file path
    :return: Trained Autoencoder anomaly detection model
    """

    # Read data
    properties_df = pd.read_json(data_filepath, orient='columns')

    # Apply MinMax scaler
    # NOTE: The dataset was not scaled for ease of business interpretation
    # Load your model
    scaler_model: MinMaxScaler = joblib.load(scaler_filepath)
    # Apply scaled transformation
    properties_scaled = scaler_model.transform(properties_df)
    properties_df = pd.DataFrame(properties_scaled, columns=properties_df.columns, index=properties_df.index)

    # Build Autoencoder
    model = AutoEncoder(output_units=x_train_scaled.shape[1])
    # configurations of model
    model.compile(loss='msle', metrics=['mse'], optimizer='adam')

    history = model.fit(
        x_train_scaled,
        x_train_scaled,
        epochs=20,
        batch_size=512,
        validation_data=(x_test_scaled, x_test_scaled)
    )
