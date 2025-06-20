"""
Author: Cesar M. Gonzalez
Find nearest properties
"""

from src.find_nearest_properties import find_nearest_properties
import pandas as pd

print('Execute Nearest properties script - properties Medellín')

# Load test dataset
print('Load inference dataset')
inference_data_filepath = './data/inference_properties.json'
properties_inference_df = pd.read_json(inference_data_filepath, lines=True)

# [0] stratum_name
# [1] propertyType
# [2] price **
# [3] rooms
# [4] baths
# [5] area **
# [6] administration_price
# [7] age
# [8] garages
# [9] gmaps_geolocations_lat
# [10] gmaps_geolocations_lng
# [11] published_year
# [12] published_month
reference_anomaly_instance = [4, 1, 530000000, 3, 2, 155.0, 360000, 1, 1, 6.1560483, -75.5880241, 2024, 4]
# Define the features used to get the closest properties
features_to_evaluate = ['stratum_name',
                        'propertyType',
                        # 'price',
                        'rooms',
                        'baths',
                        # 'area',
                        # 'administration_price',
                        'age',
                        # 'garages',
                        'gmaps_geolocations_lat',
                        'gmaps_geolocations_lng',
                        # 'published_year',
                        # 'published_month'
                        ]
n_closest_neighbors = 5
# Find Closest properties
nearest_properties, _ = find_nearest_properties(properties_data=properties_inference_df,
                                                features_to_evaluate=features_to_evaluate,
                                                r_instance=reference_anomaly_instance,
                                                n_closest_neighbors=n_closest_neighbors)
# Save results
print('Save closest properties results')
nearest_properties.to_csv('./data/nearest_properties.csv', index=False)
