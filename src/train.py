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
from torch.utils.data import DataLoader, TensorDataset, random_split


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, 5),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(5, 7),
            nn.ReLU(),
            nn.Linear(7, 10),
            nn.ReLU(),
            nn.Linear(10, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(data_filepath: str, scaler_filepath: str) -> None:
    """
    Train Autoencoder anomaly detection model
    :param data_filepath: Data file path
    :param scaler_filepath: MinMax scaler file path
    :return: Trained Autoencoder anomaly detection model
    """

    # Read data
    properties_df = pd.read_json(data_filepath, lines=True)

    # Apply MinMax scaler
    # NOTE: The dataset was not scaled for ease of business interpretation
    # Load your model
    scaler_model: MinMaxScaler = joblib.load(scaler_filepath)
    # Apply scaled transformation
    properties_scaled = scaler_model.transform(properties_df)
    properties_df = pd.DataFrame(properties_scaled, columns=properties_df.columns, index=properties_df.index)

    # Convert to PyTorch tensor
    tensor_data = torch.tensor(properties_df.values, dtype=torch.float32)

    # Split into training and validation sets
    train_size = int(0.8 * len(tensor_data))
    val_size = len(tensor_data) - train_size
    train_dataset, val_dataset = random_split(tensor_data, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Init the autoencoder Hyperparameters
    input_dim = properties_df.shape[1]
    num_epochs = 300
    learning_rate = 0.001

    # Initialize the model, loss function and optimizer
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    patience = 5

    # Training loop
    for epoch in range(num_epochs):
        for data in train_loader:
            # Forward pass
            output = model(data)
            loss = criterion(output, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
