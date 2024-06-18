"""
Author: Cesar M. Gonzalez

Train Autoencoder anomaly detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.autoencoder_model import Autoencoder, EarlyStopping


def train_autoencoder(properties_data: str) -> Autoencoder:
    """
    Train Autoencoder anomaly detection model
    :param properties_data: Properties data
    :return: Trained Autoencoder anomaly detection model
    """
    # Build Autoencoder Anomaly Detection method
    # Convert to PyTorch tensor
    tensor_data = torch.tensor(properties_data, dtype=torch.float32)

    # Split into training and validation sets
    train_size = int(0.8 * len(tensor_data))
    val_size = len(tensor_data) - train_size
    train_dataset, val_dataset = random_split(tensor_data, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Init the autoencoder Hyperparameters
    input_dim = len(properties_data[0])
    num_epochs = 300
    learning_rate = 0.001

    # Initialize the model, loss function and optimizer
    model: Autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'epoch [{epoch + 1}/{num_epochs}], loss: {train_loss:.8f}, validation loss: {val_loss:.8f}')

        # Check for early stopping
        if early_stopping.step(val_loss):
            print("Early stopping triggered")
            break

    # Return model filepath
    return model
