"""
Author: Cesar M. Gonzalez
Inference Autoencoder anomaly detection model
"""
import joblib
import os
import torch
import pandas as pd
from src.autoencoder_model import Autoencoder
from src.constants import MODEL_FILEPATH, N_FEATURES, SCALER_FILEPATH
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


def anomaly_detection_properties_inference(properties_data, r_instance: [], n_closest_anomalies: int):
    """
    Find the anomalies in the properties data
    :param properties_data: Properties data
    :param r_instance: Reference reconstructed error instance, used to find anomalies
    :param n_closest_anomalies: Number of closest anomalies to filter
    :return: Trained Autoencoder anomaly detection model and report dict
    """
    print('Identify the closest properties anomalies')
    print(f'Reference instance: {r_instance}')
    print(f'Get the closest {n_closest_anomalies} properties anomalies')

    # Check inference conditions
    assert os.path.exists(MODEL_FILEPATH), 'Anomaly detection model file does not exist! execute training pipeline '
    assert len(r_instance) == N_FEATURES, 'The reference instance is not the correct number of features!'
    assert [inst for inst in r_instance if -1 <= inst <= 1], 'Reference instance values must be in range [0, 1]'

    # Preprocess inference, Apply normalization
    scaler_model: MinMaxScaler = joblib.load(SCALER_FILEPATH)
    # Apply scaler transformation Train and Test datasets
    properties_data_scaled = scaler_model.transform(properties_data)

    # Transform data to tensors
    tensor_data = torch.tensor(properties_data_scaled, dtype=torch.float32)

    # Apply anomaly detection model
    model = Autoencoder(N_FEATURES)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()
    reconstructed_data = model(tensor_data)

    # Calculate the reconstructed error for each feature
    reconstructed_data_narray = reconstructed_data.detach().cpu().numpy()
    reconstruction_error = properties_data_scaled - reconstructed_data_narray

    # Create a Nearest Neighbors instance (finding nearest neighbors)
    nn = NearestNeighbors(n_neighbors=n_closest_anomalies)
    # Fit the model on the dataset
    nn.fit(reconstruction_error)
    # Find the nearest neighbors
    distances, indices = nn.kneighbors([r_instance])

    # Build the reconstructed df
    reconstructed_properties = scaler_model.inverse_transform(reconstructed_data_narray)
    reconstructed_properties_df = pd.DataFrame(reconstructed_properties, columns=properties_data.columns,
                                               index=properties_data.index)

    print('Properties anomalies detected')
    # Get anomalies from the original data
    return properties_data.iloc[indices[0]], reconstructed_properties_df.iloc[indices[0]], distances
