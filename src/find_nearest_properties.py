"""
Author: Cesar M. Gonzalez
Inference Autoencoder anomaly detection model
"""
import joblib
import os
import pandas as pd
from src.constants import MODEL_FILEPATH, SCALER_FILEPATH, VALID_FEATURES
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


def find_nearest_properties(properties_data, features_to_evaluate, r_instance: [], n_closest_neighbors: int):
    """
    Find the closest  properties based on reference
    :param properties_data: Properties data
    :param features_to_evaluate: Properties features to evaluate
    :param r_instance: Reference property, used to find the closest properties around
    :param n_closest_neighbors: Number of closed neighbors to filter
    :return: Trained Autoencoder anomaly detection model and report dict
    """
    print('Identify the closest properties based on reference')
    print(f'Reference instance: {r_instance}')
    print(f'Get the closest {n_closest_neighbors} properties')

    # Check inference conditions
    assert os.path.exists(MODEL_FILEPATH), 'Anomaly detection model file does not exist! execute training pipeline!'
    assert len(features_to_evaluate) == len(set(features_to_evaluate)), 'Features list has duplicate values!'
    assert set(features_to_evaluate).issubset(set(VALID_FEATURES)), 'There are invalid features to evaluate!'

    # Preprocess inference, Apply normalization
    scaler_model: MinMaxScaler = joblib.load(SCALER_FILEPATH)
    # Apply scaler transformation Train and Test datasets
    properties_data_scaled = scaler_model.transform(properties_data)
    # Apply scaler transformation to reference instance
    property_reference_scaled = scaler_model.transform(pd.DataFrame([r_instance], columns=properties_data.columns))

    # Filter by relevant columns
    properties_data_scaled_df = pd.DataFrame(properties_data_scaled, columns=properties_data.columns)
    properties_data_scaled_df = properties_data_scaled_df[features_to_evaluate]
    property_reference_scaled_df = pd.DataFrame(property_reference_scaled, columns=properties_data.columns)
    property_reference_scaled_df = property_reference_scaled_df[features_to_evaluate]

    # Create a Nearest Neighbors instance (finding nearest neighbors)
    nn = NearestNeighbors(n_neighbors=n_closest_neighbors)
    # Fit the model on the dataset
    nn.fit(properties_data_scaled_df.values)
    # Find the nearest neighbors
    distances, indices = nn.kneighbors([property_reference_scaled_df.values[0]])

    print('Closet Properties detected')
    # Get anomalies from the original data
    return properties_data.iloc[indices[0]], distances
