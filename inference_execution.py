"""
Author: Cesar M. Gonzalez

Inference Autoencoder anomaly detection model
"""

from src.inference import anomaly_detection_properties_inference
import pandas as pd

print('Execute Anomaly detection model inference - properties Medellín')

# Load test dataset
print('Load inference dataset')
inference_data_filepath = './data/inference_properties.json'
properties_inference_df = pd.read_json(inference_data_filepath, lines=True)

# The reference anomaly instance contains the values where we want to find the anomaly based on the reconstruction error
# As you can se we desire find the properties with lower price (price = -1) and big area (area = 1)
# [0] stratum_name
# [1] propertyType
# [2] price
# [3] rooms
# [4] baths
# [5] area
# [6] administration_price
# [7] age
# [8] garages
# [9] gmaps_geolocations_lat
# [10] gmaps_geolocations_lng
# [11] published_year
# [12] published_month
reference_anomaly_instance = [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
n_close_anomalies = 5
anomalies_df, r_anomalies_df, _ = anomaly_detection_properties_inference(properties_data=properties_inference_df,
                                                                         r_instance=reference_anomaly_instance,
                                                                         n_close_anomalies=n_close_anomalies)
# Save results
print('Save properties anomalies results')
anomalies_df.to_csv('./data/properties_anomalies.csv', index=False)
r_anomalies_df.to_csv('./data/properties_anomalies_reconstructed.csv', index=False)

