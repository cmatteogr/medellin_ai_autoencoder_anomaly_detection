from src.train import train_autoencoder

data_filepath = './data/train_properties.json'
scaler_filepath = './src/preprocess_scaler_model.pkl'
train_autoencoder(data_filepath, scaler_filepath)