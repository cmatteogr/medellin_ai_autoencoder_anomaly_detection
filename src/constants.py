# Autoencoder Model filepath
MODEL_FILEPATH = './src/autoencoder_anomaly_detection_model.pth'
# Scaler Model filepath
SCALER_FILEPATH = './src/preprocess_scaler_model.pkl'
# Number of dataset features
N_FEATURES = 13
# Valid properties features
VALID_FEATURES = ('stratum_name', 'propertyType', 'price', 'rooms', 'baths', 'area', 'administration_price', 'age',
                  'garages', 'gmaps_geolocations_lat', 'gmaps_geolocations_lng', 'published_year', 'published_month')
