"""
Author: Cesar M. Gonzalez

Train Autoencoder anomaly detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.constants import N_FEATURES
from src.autoencoder_model import Autoencoder, EarlyStopping
import optuna


def train_autoencoder(properties_data: str, train_size_percentage=0.8, batch_size=32) -> (Autoencoder, dict):
    """
    Train Autoencoder anomaly detection model
    :param properties_data: Properties data
    :param train_size_percentage: Train size percentage, remaining is validation size
    :param batch_size: Batch size
    :return: Trained Autoencoder anomaly detection model and report dict
    """
    print('# Start training autoencoder anomaly detection model')

    # Check input arguments
    assert 0.7 <= train_size_percentage < 1, 'Train size percentage should be between 0.7 and 1.'
    assert 1 <= batch_size <= 256, 'Batch size should be between 1 and 124.'

    # Build Autoencoder Anomaly Detection method
    # Convert to PyTorch tensor
    tensor_data = torch.tensor(properties_data, dtype=torch.float32)

    # Split into training and validation sets
    print('Split dataset into train and validation set')
    train_size = int(train_size_percentage * len(tensor_data))
    val_size = len(tensor_data) - train_size
    train_dataset, val_dataset = random_split(tensor_data, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Init the autoencoder Hyperparameters
    num_epochs = 300
    early_stopping_patience = 15

    # Build the model tunner using optuna
    print('Build Autoencoder anomaly detection model')

    def train_model(trial):
        # Init the Hyperparameters to change
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # Init the Autoencoder, loss function metric and optimizer
        model: Autoencoder = Autoencoder(N_FEATURES)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Early stopping is added to avoid overfitting
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training loop
        print('Training autoencoder anomaly detection model')
        reconstruction_error = float('inf')
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
            # Calculate train batch loss
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            # Calculate validation batch loss
            val_loss /= len(val_loader)
            print(f'epoch [{epoch + 1}/{num_epochs}], loss: {train_loss:.8f}, validation loss: {val_loss:.8f}')

            # Save last mse
            reconstruction_error = val_loss

            # Check for early stopping to avoid overfitting
            if early_stopping.step(val_loss):
                print("Early stopping triggered")
                break

        return reconstruction_error

    # Execute optuna optimizer study
    study = optuna.create_study(direction='minimize')
    study.optimize(train_model, n_trials=50)
    # Get Best parameters
    best_params = study.best_params
    print('Best params: {}'.format(best_params))

    # Retrain the best model
    # NOTE: The retraining is done from scratch using best hyperparameters to ensure the values benefit the model
    # Init the autoencoder Hyperparameters. The number of epochs is increased
    num_epochs = 500
    early_stopping_patience = 15

    # Init the Autoencoder, loss function metric and optimizer
    model: Autoencoder = Autoencoder(N_FEATURES)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    # Early stopping is added to avoid overfitting
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    # Training loop
    print(f'Training autoencoder anomaly detection model. Best params: {best_params}')
    reconstruction_error = float('inf')
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
        # Calculate train batch loss
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        # Calculate validation batch loss
        val_loss /= len(val_loader)
        print(f'epoch [{epoch + 1}/{num_epochs}], loss: {train_loss:.8f}, validation loss: {val_loss:.8f}')

        # Save last mse
        reconstruction_error = val_loss

        # Check for early stopping to avoid overfitting
        if early_stopping.step(val_loss):
            print("Early stopping triggered")
            break

    print(f'Reconstruction error, best score: {reconstruction_error:.8f}')
    # Build train report
    print('Create training report dict')
    report_dict = {
        'reconstruction_error': float(reconstruction_error)
    }

    print('End training autoencoder anomaly detection model')
    # Return model filepath
    return model, report_dict
