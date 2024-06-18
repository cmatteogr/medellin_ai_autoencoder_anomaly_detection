"""
Author: Cesar M. Gonzalez

Training Autoencoder anomaly detection model pipeline
"""

from src.train import train_autoencoder
from src.test import test_autoencoder
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

# Read datasets
# Load train dataset
train_data_filepath = './data/train_properties.json'
properties_train_df = pd.read_json(train_data_filepath, lines=True)
# Load test dataset
test_data_filepath = './data/test_properties.json'
properties_test_df = pd.read_json(test_data_filepath, lines=True)

# Preprocess data
# NOTE: The dataset were not scaled for ease of business interpretation.

# Apply MinMax scaler
# Load scaler model
scaler_filepath = './src/preprocess_scaler_model.pkl'
scaler_model: MinMaxScaler = joblib.load(scaler_filepath)
# Apply scaler transformation Train and Test datasets
properties_train_scaled = scaler_model.transform(properties_train_df)
properties_test_scaled = scaler_model.transform(properties_test_df)

# Init Autoencoder Training


autoencoder_model = train_autoencoder(properties_train_scaled)

# Test Autoencoder performance

test_autoencoder(autoencoder_model, properties_test_scaled)
