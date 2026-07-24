# How to find Pearls at the bottom of the Sea

![pearl_bottom_sea](https://github.com/cmatteogr/medellin_ai_autoencoder_anomaly_detection/assets/138587358/86296620-59b0-466f-8f1f-e574941373ba)

Finding anomalies in a data set does not always represent unwanted values, sometimes they are opportunities, but finding them can be a challenge in the face of the immensity of the data. An autoencoder-based anomaly detection model allows finding these opportunities (anomalies) with the desired characteristics (pearls) and may be a more robust solution to this task than linear anomaly detection models.

The data set used contains properties for sale in Medellín (2021-2024). The goal is to find the good opportunities in the real state Medellín market.

## Workshop
An interactive workshop notebook is available in `notebooks/`, designed to run end to end on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmatteogr/medellin_ai_autoencoder_anomaly_detection/blob/main/notebooks/workshop_autoencoder_anomaly_detection.ipynb)

It rebuilds the project step by step and extends it with a single configurable autoencoder whose regularization terms and latent-space transformations can be switched on, off, and combined:

* **Denoising** (gaussian / masking input corruption), **Sparse** (L1 and KL-divergence flavours), **Variational** (β-VAE), **Contractive** (Jacobian penalty via the Hutchinson estimator), **Hyperspherical** (unit-sphere latent) and **Deep SVDD** (hypersphere compactness) — plus hybrids such as denoising + sparse + variational.
* The *good anomaly* search is explained as a **direction in reconstruction-error space**, including why the repository's nearest-neighbour query is equivalent to maximising `e·d - ½‖e‖²`.
* A **clustering** section groups the signed error vectors so the recurring *kinds* of anomaly emerge from the data instead of being assumed.

## Setup
Use python 3.12 and install the dependecies using requirements.txt
```bash
pip install -r requirements.txt
```

## Usage
The project has two main scripts:
* train_pipeline: Pipeline to train the properties anomaly detection model (based on Autoencoder)
```bash
python train_pipeline.py
```
* inference_execution: Execute the properties anomaly detection model, it returns the anomalies csv files and the reconstructed anomalies csv file.
```bash
python inference_execution.py
```
* find_nearest_properties: Used it to find similar properties based on their features in order to compare the anomalies.
```bash
python find_nearest_properties.py
```
