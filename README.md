# How to find pearls at the bottom of the ocean

![pearl_bottom_sea](https://github.com/cmatteogr/medellin_ai_autoencoder_anomaly_detection/assets/138587358/86296620-59b0-466f-8f1f-e574941373ba)

Finding anomalies in a data set does not always represent unwanted values, sometimes they are opportunities, but finding them can be a challenge in the face of the immensity of the data. An autoencoder-based anomaly detection model allows finding these opportunities (anomalies) with the desired characteristics (pearls) and may be a more robust solution to this task than linear anomaly detection models.

The dataset used contains Medellin properties on sale (2021-2024). The goal is find the good oppotunities in the market

## Setup
Use python 3.12 and install the dependecies using requirements.txt

# Usage
The project has two main scripts:
* train_pipeline: Pipeline to train the properties anomaly detection model (based on Autoencoder)
* inference_execution: Execute the properties anomaly detection model, it returns the anomalies csv files and the reconstructed anomalies csv file.
