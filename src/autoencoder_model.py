"""
Author: Cesar M. Gonzalez
Autoencoder anomaly detection model
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 7),
            nn.LeakyReLU(),
            nn.Linear(7, 5),
            nn.LeakyReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(5, 7),
            nn.LeakyReLU(),
            nn.Linear(7, 10),
            nn.LeakyReLU(),
            nn.Linear(10, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Autoencoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False
