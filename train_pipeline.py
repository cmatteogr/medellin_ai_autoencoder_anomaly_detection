"""
Author: Cesar M. Gonzalez

Training Autoencoder anomaly detection model pipeline
"""
from src.constants import SCALER_FILEPATH, MODEL_FILEPATH
from src.train import train_autoencoder
from src.test import test_autoencoder
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import os
import json
import sys
import torch

# Collect data
print('Read Train and Test data')
# Load train dataset
train_data_filepath = './data/train_properties.json'
properties_train_df = pd.read_json(train_data_filepath, lines=True)
# Load test dataset
test_data_filepath = './data/test_properties.json'
properties_test_df = pd.read_json(test_data_filepath, lines=True)

# Preprocess data
print('Preprocess Train and Test data')
# NOTE: The datasets were not scaled for ease of business interpretation. The normalization is done in runtime
# Load scaler model and apply MinMax scaler
scaler_model: MinMaxScaler = joblib.load(SCALER_FILEPATH)
# Apply scaler transformation Train and Test datasets
properties_train_scaled = scaler_model.transform(properties_train_df)
properties_test_scaled = scaler_model.transform(properties_test_df)

# Train autoencoder for anomaly detection model
print('Train anomaly detection model (Autoencoder)')
autoencoder_model, train_report_dict = train_autoencoder(properties_train_scaled, batch_size=32)

# Test Autoencoder performance
print('Test anomaly detection model (Autoencoder)')
test_report_dict = test_autoencoder(autoencoder_model, properties_test_scaled)

# Evaluate if deploy the model
reconstruction_error_threshold = 0.007
if test_report_dict['reconstruction_error'] > reconstruction_error_threshold:
    print(f'The model doesn\'t have a good reconstruction error {test_report_dict["reconstruction_error"]}')
    print('Execute again or update hyperparameters if needed')
    sys.exit()

# If previous model already exist the read report and compare reconstruction error
deploy_model = True
report_filepath = './src/autoencoder_anomaly_detection_report.json'
if os.path.exists(report_filepath):
    # Read report and compare reconstruction error
    with open(report_filepath, 'r') as json_file:
        model_report = json.load(json_file)

    # Compare reconstruction error, define if update current model
    deploy_model = model_report['reconstruction_error'] > test_report_dict['reconstruction_error']
    print(f'Trained model has better performance than current deployed model: {deploy_model}')
else:
    # Create First model version
    print('None deployed model found, deploy trained model')
    deploy_model = True

# Update report and model if needed
if deploy_model:
    # Save the report to a JSON file
    with open(report_filepath, 'w') as json_file:
        json.dump(test_report_dict, json_file)
    # Save the trained model
    torch.save(autoencoder_model.state_dict(), MODEL_FILEPATH)