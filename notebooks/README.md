# Workshop notebook

`workshop_autoencoder_anomaly_detection.ipynb` is a self-contained, Colab-ready
walkthrough of this project. It rebuilds the pipeline step by step and extends it
with a **single configurable autoencoder** whose regularization terms and
latent-space transformations can be switched on, off, and combined.

## Run it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmatteogr/medellin_ai_autoencoder_anomaly_detection/blob/main/notebooks/workshop_autoencoder_anomaly_detection.ipynb)

The first cell clones this repository and loads `data/*.json`, so nothing needs to
be uploaded. CPU is enough; the whole notebook runs in a few minutes.

## What it covers

1. The Medellín real-estate dataset (13 features, 17.5k listings).
2. Why a `MinMaxScaler` and a `Sigmoid` output layer belong together.
3. The vanilla autoencoder from `src/autoencoder_model.py`.
4. A **regularization zoo** built as composable switches on one class:
   - denoising (gaussian / masking input corruption),
   - sparse (L1 and KL-divergence flavours),
   - variational (β-VAE, with the reparameterization trick),
   - contractive (Jacobian penalty via the Hutchinson estimator),
   - hyperspherical (unit-sphere latent),
   - Deep SVDD (hypersphere compactness),
   - and hybrids such as denoising + sparse + variational.
   An **interactive dropdown** then illustrates how each variant works vs the
   vanilla autoencoder — the mechanism, plus its advantage and disadvantage for
   finding good anomalies.
5. Finding **good anomalies** as a *direction* in reconstruction-error space,
   including why the repository's nearest-neighbour query is equivalent to
   maximising `e·d − ½‖e‖²`.
6. **Clustering** the signed error vectors so the recurring *kinds* of anomaly
   emerge from the data instead of being assumed.
7. An independent comparables check to sanity-test the "pearls".
8. Exercises.

## Regenerating the notebook

The `.ipynb` is generated from `build_notebook.py` (building it programmatically
keeps the cell sources readable and diffable). After editing that script:

```bash
python notebooks/build_notebook.py notebooks
```

To re-validate that every cell still runs end to end:

```bash
pip install nbconvert ipykernel matplotlib
jupyter nbconvert --to notebook --execute \
  notebooks/workshop_autoencoder_anomaly_detection.ipynb \
  --output /tmp/executed.ipynb
```