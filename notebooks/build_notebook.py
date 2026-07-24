"""
Generates notebooks/workshop_autoencoder_anomaly_detection.ipynb

Building the .ipynb programmatically (instead of hand-writing JSON) keeps the
cell sources readable and lets us re-generate after edits.
"""
import json
import os
import sys

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n"),
        }
    )


# --------------------------------------------------------------------------
# 0. Title
# --------------------------------------------------------------------------
md(r"""
# How to find Pearls at the bottom of the Sea
### Autoencoders, regularization, and the search for *good* anomalies

![pearl_bottom_sea](https://github.com/cmatteogr/medellin_ai_autoencoder_anomaly_detection/assets/138587358/86296620-59b0-466f-8f1f-e574941373ba)

An anomaly is just *something the model did not expect*. Most anomaly detection
treats that as bad news — fraud, a broken sensor, a defect. But an unexpected
value can also be an **opportunity**: an apartment that is much cheaper, or much
bigger, than everything else that looks like it.

This workshop uses **17,544 properties for sale in Medellín (2021–2024)** and asks
a very concrete question:

> *Which listings are priced far below what their own characteristics predict?*

We answer it with autoencoders — building **one configurable model** whose
regularization terms and latent transforms can be switched on, off, and
**combined**. Vanilla, sparse, denoising, variational, and every hybrid are the
same class with different settings.
""")

md(r"""
## Agenda

| # | Section | What you learn |
|---|---------|----------------|
| 1 | Setup | Clone the repo, install, seed everything |
| 2 | The data | 13 features of the Medellín real-estate market |
| 3 | Preprocessing | Why `MinMaxScaler` + a `Sigmoid` output belong together |
| 4 | The vanilla autoencoder | The baseline from `src/autoencoder_model.py` |
| 5 | **The regularization zoo** | Sparse / denoising / variational / contractive / hyperspherical / SVDD — as composable terms |
| 6 | Variant gallery | Train them, compare them, and an **interactive panel** for each variant's impact vs vanilla |
| 7 | **Good anomalies** | The *direction* trick in reconstruction-error space |
| 8 | Clustering the error space | Discovering *kinds* of anomaly, not just a ranking |
| 9 | Sanity check | Do the pearls survive a comparison with their neighbours? |
| 10 | Exercises | Your turn |

**Runtime:** CPU is enough. The whole notebook runs in a few minutes.
""")

# --------------------------------------------------------------------------
# 1. Setup
# --------------------------------------------------------------------------
md(r"""
---
## 1. Setup

The dataset and the reference implementation both live in the project repository,
so the first cell clones it and moves into it. Re-running the cell is safe.
""")

code(r'''
import os
import subprocess
import sys

REPO_URL = "https://github.com/cmatteogr/medellin_ai_autoencoder_anomaly_detection.git"
REPO_DIR = "medellin_ai_autoencoder_anomaly_detection"
MARKER = "data/train_properties.json"


def repo_root_from_here():
    """Walk up from the working directory looking for the checked-out repo.

    Covers the case where the notebook is opened from notebooks/ inside a local
    clone -- without this it would try to clone the repo into itself.
    """
    path = os.getcwd()
    for _ in range(4):
        if os.path.exists(os.path.join(path, MARKER)):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


root = repo_root_from_here()
if root is None:
    # Colab / any fresh machine: fetch the repo.
    if not os.path.isdir(REPO_DIR):
        print("Cloning the workshop repository...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    root = os.path.abspath(REPO_DIR)

os.chdir(root)

assert os.path.exists(MARKER), "Could not find the dataset -- did the clone succeed?"
print("Working directory:", os.getcwd())
print("Data files:", sorted(os.listdir("data")))
''')

code(r'''
# Colab already ships these, so this is normally a no-op. It only does real work
# on a bare local environment.
IN_COLAB = "google.colab" in sys.modules

try:
    import torch, pandas, sklearn, matplotlib, ipywidgets  # noqa: F401
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch", "pandas", "scikit-learn", "matplotlib", "ipywidgets"],
        check=True,
    )

print("Running in Colab:", IN_COLAB)
''')

code(r'''
import json
import warnings
from dataclasses import dataclass, replace

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make a run reproducible across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", DEVICE)
''')

md(r"""
### A shared chart style

Every plot uses the same palette, so you compare *data*, not decoration. Colours
are assigned by the **job** they do:

- **categorical** (identity — train vs. validation) — a fixed hue order;
- **sequential** (magnitude — price) — one hue, light to dark;
- **diverging** (polarity — a signed error) — two hues, neutral grey midpoint.
""")

code(r'''
# Categorical slots, in fixed order (never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Sequential: one hue (blue), light -> dark. For magnitude.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# Diverging: blue <-> red with a neutral grey midpoint. For signed values.
DIV_BR = LinearSegmentedColormap.from_list(
    "div_br",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
     "#f7c4c4", "#e34948", "#b02b2b", "#7d1c1c"],
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "figure.dpi": 110,
})


def tidy(ax, ygrid_only=True):
    """Recessive chrome: drop the top/right spines, keep one grid direction."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid_only:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    return ax


print("Chart style ready.")
''')

# --------------------------------------------------------------------------
# 2. The data
# --------------------------------------------------------------------------
md(r"""
---
## 2. The data

Three splits ship with the repository, all in JSON-lines format:

- `train_properties.json` — what the autoencoder learns "normal" from;
- `test_properties.json` — held out, used to decide whether a model is good enough;
- `inference_properties.json` — the pool we actually hunt for pearls in.

Note the datasets are stored **unscaled**, on purpose: 430,000,000 COP is readable,
`0.31` is not. Scaling happens at runtime.
""")

code(r'''
properties_train_df = pd.read_json("./data/train_properties.json", lines=True)
properties_test_df = pd.read_json("./data/test_properties.json", lines=True)
properties_inference_df = pd.read_json("./data/inference_properties.json", lines=True)

FEATURES = list(properties_train_df.columns)
N_FEATURES = len(FEATURES)

print(f"train     {properties_train_df.shape}")
print(f"test      {properties_test_df.shape}")
print(f"inference {properties_inference_df.shape}")
print(f"\n{N_FEATURES} features:")
for i, f in enumerate(FEATURES):
    print(f"  [{i:2d}] {f}")

properties_train_df.head()
''')

md(r"""
### What the 13 features mean

| # | Feature | Meaning |
|---|---------|---------|
| 0 | `stratum_name` | Colombian socio-economic stratum, 0–6 (a strong price proxy) |
| 1 | `propertyType` | 1 = apartment, 0 = house |
| 2 | `price` | Asking price, COP |
| 3 | `rooms` | Bedrooms |
| 4 | `baths` | Bathrooms |
| 5 | `area` | m² |
| 6 | `administration_price` | Monthly building fee, COP |
| 7 | `age` | Age bucket, 0 = new |
| 8 | `garages` | Parking spots |
| 9 | `gmaps_geolocations_lat` | Latitude |
| 10 | `gmaps_geolocations_lng` | Longitude |
| 11 | `published_year` | Listing year |
| 12 | `published_month` | Listing month |

Two things matter for what follows. First, **there is no label** — nobody told us
which listings are bargains, so this is strictly unsupervised. Second, the features
are **heavily correlated** (stratum, location, area and price all move together),
and that correlation is exactly what an autoencoder compresses. A listing that
*breaks* the correlation is the one we want.
""")

code(r'''
fig, axes = plt.subplots(3, 5, figsize=(15, 7.5))
for ax, feature in zip(axes.ravel(), FEATURES):
    ax.hist(properties_train_df[feature], bins=40,
            color=PALETTE[0], edgecolor="white", linewidth=0.4)
    ax.set_title(feature, fontsize=9)
    tidy(ax)
for ax in axes.ravel()[N_FEATURES:]:
    ax.set_visible(False)
fig.suptitle("Feature distributions — training split", x=0.005, ha="left",
             fontsize=13, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
''')

code(r'''
# The relationship the autoencoder has to learn: price is largely explained by
# area and stratum. The interesting listings are the ones far off this cloud.
fig, ax = plt.subplots(figsize=(7.5, 5))
sc = ax.scatter(
    properties_train_df["area"],
    properties_train_df["price"] / 1e6,
    c=properties_train_df["stratum_name"],
    cmap=SEQ_BLUE, s=9, alpha=0.55, linewidths=0,
)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("stratum", color=INK_2, fontsize=9)
cbar.outline.set_visible(False)
ax.set_xlabel("area (m²)")
ax.set_ylabel("price (millions COP)")
ax.set_title("Price vs. area, coloured by stratum")
tidy(ax, ygrid_only=False)
plt.show()
''')

# --------------------------------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------------------------------
md(r"""
---
## 3. Preprocessing — scaling that matches the output layer

The decoder ends in a `Sigmoid`, which can only emit values in $(0, 1)$. So we
scale the inputs into $[0, 1]$ with a **MinMax** scaler — target range and model
range agree:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$

A `StandardScaler` (unbounded) would ask a `Sigmoid` to reproduce values it can
never reach. **Match the output activation to the target range** — a classic
autoencoder bug. We reuse the scaler the repo already fitted
(`src/preprocess_scaler_model.pkl`) so the numbers here match the reference code.
""")

code(r'''
scaler: MinMaxScaler = joblib.load("./src/preprocess_scaler_model.pkl")

X_train = scaler.transform(properties_train_df).astype(np.float32)
X_test = scaler.transform(properties_test_df).astype(np.float32)
X_infer = scaler.transform(properties_inference_df).astype(np.float32)

print("scaled train range: [%.3f, %.3f]" % (X_train.min(), X_train.max()))
print("scaled test  range: [%.3f, %.3f]" % (X_test.min(), X_test.max()))

pd.DataFrame({
    "feature": FEATURES,
    "data_min": scaler.data_min_,
    "data_max": scaler.data_max_,
}).set_index("feature")
''')

code(r'''
# Tensors + loaders, reused by every model we train below.
BATCH_SIZE = 256
VAL_FRACTION = 0.2

X_train_t = torch.tensor(X_train)
X_test_t = torch.tensor(X_test)
X_infer_t = torch.tensor(X_infer)

set_seed()
n_val = int(VAL_FRACTION * len(X_train_t))
perm = torch.randperm(len(X_train_t))
val_idx, train_idx = perm[:n_val], perm[n_val:]

X_fit_t, X_val_t = X_train_t[train_idx], X_train_t[val_idx]

train_loader = DataLoader(TensorDataset(X_fit_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False)

print(f"fit: {len(X_fit_t)}  |  val: {len(X_val_t)}  |  test: {len(X_test_t)}")
''')

# --------------------------------------------------------------------------
# 4. Vanilla AE
# --------------------------------------------------------------------------
md(r"""
---
## 4. The vanilla autoencoder

An autoencoder is two functions trained together:

$$z = f_\theta(x) \qquad \hat{x} = g_\phi(z) \qquad
\mathcal{L} = \lVert x - \hat{x} \rVert^2$$

The trick is the **bottleneck**: $z$ has only 5 dimensions while $x$ has 13. The
network cannot memorise, so it is forced to learn the *structure* of the Medellín
market — "stratum 6 near El Poblado with 100 m² costs about this much".

```
13 ──► 10 ──► 7 ──► [ 5 ] ──► 7 ──► 10 ──► 13
   encoder        latent        decoder
```

**Why this finds anomalies.** Trained on the majority, the model reconstructs a
normal listing accurately. A listing that breaks the market's rules gets
reconstructed as *what it should have been* — and that gap is the signal.

This is the architecture from `src/autoencoder_model.py`, reproduced so the
workshop is self-contained.
""")

code(r'''
class Autoencoder(nn.Module):
    """The baseline model from src/autoencoder_model.py."""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10), nn.LeakyReLU(),
            nn.Linear(10, 7), nn.LeakyReLU(),
            nn.Linear(7, 5), nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 7), nn.LeakyReLU(),
            nn.Linear(7, 10), nn.LeakyReLU(),
            nn.Linear(10, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class EarlyStopping:
    """Stop when validation loss stops improving, to avoid overfitting."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.counter = float("inf"), 0

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter = loss, 0
        else:
            self.counter += 1
        return self.counter >= self.patience


print(Autoencoder(N_FEATURES))
''')

code(r'''
def train_vanilla(epochs=150, lr=1e-3, patience=15):
    set_seed()
    model = Autoencoder(N_FEATURES).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            loss = criterion(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                total += criterion(model(batch), batch).item()
        val_loss = total / len(val_loader)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if (epoch + 1) % 25 == 0:
            print(f"epoch {epoch + 1:3d}/{epochs}  train {train_loss:.6f}  val {val_loss:.6f}")
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, history


vanilla_model, vanilla_history = train_vanilla()
''')

md(r"""
> **Note on hyperparameters.** `src/train.py` wraps this loop in an
> [Optuna](https://optuna.org/) study that searches the learning rate over 50
> trials and then retrains from scratch with the winner. We skip the search here
> to keep the workshop interactive — but retraining from scratch after the search,
> rather than reusing the best trial's weights, is the right instinct and worth
> copying.
""")

code(r'''
fig, ax = plt.subplots(figsize=(7.5, 4.2))
epochs = range(1, len(vanilla_history["train"]) + 1)
ax.plot(epochs, vanilla_history["train"], color=PALETTE[0], label="train")
ax.plot(epochs, vanilla_history["val"], color=PALETTE[1], label="validation")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Vanilla autoencoder — reconstruction loss")
ax.legend(loc="upper right")
tidy(ax)
plt.show()
''')

code(r'''
@torch.no_grad()
def reconstruct(model, X_t):
    """Return reconstructions as a numpy array."""
    model.eval()
    out = model(X_t.to(DEVICE))
    if isinstance(out, dict):
        out = out["x_hat"]
    return out.cpu().numpy()


X_test_hat = reconstruct(vanilla_model, X_test_t)
per_row_mse = ((X_test - X_test_hat) ** 2).mean(axis=1)
per_feature_mae = np.abs(X_test - X_test_hat).mean(axis=0)

print(f"test reconstruction MSE: {per_row_mse.mean():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))

axes[0].hist(per_row_mse, bins=60, color=PALETTE[0], edgecolor="white", linewidth=0.4)
axes[0].set_yscale("log")
axes[0].set_xlabel("per-listing MSE")
axes[0].set_ylabel("listings (log scale)")
axes[0].set_title("Most listings reconstruct well — the tail is where pearls hide")
tidy(axes[0])

order = np.argsort(per_feature_mae)
axes[1].barh([FEATURES[i] for i in order], per_feature_mae[order],
             color=PALETTE[0], height=0.62)
for i, v in enumerate(per_feature_mae[order]):
    axes[1].text(v, i, f"  {v:.3f}", va="center", fontsize=8, color=INK_2)
axes[1].set_xlabel("mean absolute error (scaled units)")
axes[1].set_title("Which features are hardest to reconstruct?")
axes[1].xaxis.grid(True)
axes[1].yaxis.grid(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.tight_layout()
plt.show()
''')

md(r"""
The left histogram is the whole idea in one picture: a dense peak the model
reproduces almost perfectly, and a long right tail it cannot. **The tail is the
hunting ground.**

But one MSE number collapses 13 features into one and hides *which direction* a
listing is unusual — an unexpectedly **expensive** listing and an unexpectedly
**cheap** one can share the same MSE. Section 7 fixes this.
""")

# --------------------------------------------------------------------------
# 5. The zoo
# --------------------------------------------------------------------------
md(r"""
---
## 5. The regularization zoo

A plain autoencoder has one weakness: given enough capacity it learns to *copy* its
input — reconstructing the anomalies too — and the signal vanishes. Each named
variant is a different way to stop that.

The key idea: **these are not different architectures. They are extra terms on the
same loss, plus optional transforms of the latent vector.** So we build them as
switches on one model.

$$\mathcal{L} = \underbrace{\lVert x - \hat{x}\rVert^2}_{\text{reconstruction}}
+ \lambda_{1}\underbrace{\lVert z\rVert_1}_{\text{sparse}}
+ \beta\underbrace{D_{KL}\!\left(q(z|x)\,\Vert\,\mathcal{N}(0,I)\right)}_{\text{variational}}
+ \lambda_{c}\underbrace{\lVert J_f(x)\rVert_F^2}_{\text{contractive}}
+ \lambda_{s}\underbrace{\lVert z - c\rVert^2}_{\text{SVDD}}$$

| Variant | What it changes | Why it helps |
|---|---|---|
| **Denoising** | Corrupts the input; scores against the *clean* target | Can't copy its input, so it learns what the data *should* look like |
| **Sparse (L1)** | Penalises latent activations, $\lambda\sum_j\lvert z_j\rvert$ | Few units fire per listing — a compact code |
| **Sparse (KL)** | Pushes each unit's mean activation toward a small $\rho$ | Same goal; needs a `sigmoid` latent (activations in $(0,1)$) |
| **Variational** | Encoder emits a *distribution* $\mu,\log\sigma^2$; sample, KL to $\mathcal{N}(0,I)$ | A smooth latent where "how unlikely is this?" is answerable |
| **Contractive** | Penalises the encoder's Jacobian | Latent barely moves when the input jitters — robust features |
| **Hyperspherical** | Projects $z$ onto the unit sphere | Keeps only *direction*; similarity becomes an angle |
| **Deep SVDD** | Pulls latents toward a fixed centre $c$ | $\lVert z-c\rVert$ becomes a second, independent anomaly score |

Two tricks make two of these cheap:

- **Variational** samples with $z=\mu+\sigma\odot\epsilon$ (the *reparameterization
  trick*) so the sampling stays differentiable.
- **Contractive** skips the full Jacobian with the *Hutchinson estimator*:
  $\mathbb{E}_v\lVert J^\top v\rVert^2=\lVert J\rVert_F^2$ — one extra backward pass
  instead of five.
""")

code(r'''
@dataclass
class AEConfig:
    """Every knob of the configurable autoencoder.

    The default values reproduce the vanilla model from src/autoencoder_model.py
    exactly, so every variant below is a delta from a known baseline.
    """

    name: str = "vanilla"

    # ---- architecture ----
    input_dim: int = N_FEATURES
    hidden_dims: tuple = (10, 7)
    latent_dim: int = 5
    latent_activation: str = "leaky_relu"   # "leaky_relu" | "sigmoid" | "none"

    # ---- latent-space transformation ----
    variational: bool = False               # VAE: encode a distribution, not a point
    hyperspherical: bool = False            # project the latent onto the unit sphere

    # ---- input corruption (denoising) ----
    noise_type: str = "none"                # "none" | "gaussian" | "masking"
    noise_level: float = 0.0

    # ---- regularization weights (0.0 = term switched off) ----
    l1_latent: float = 0.0                  # sparse, L1 on activations
    kl_sparsity: float = 0.0                # sparse, KL to target rho
    sparsity_rho: float = 0.05
    beta: float = 0.0                       # VAE KL weight (beta-VAE)
    contractive: float = 0.0                # Jacobian penalty
    svdd: float = 0.0                       # Deep SVDD compactness
    weight_decay: float = 0.0               # plain L2 on weights

    # ---- training ----
    lr: float = 1e-3
    epochs: int = 200      # ~20 s per variant on a Colab CPU
    patience: int = 15

    def __post_init__(self):
        if self.kl_sparsity > 0 and self.latent_activation != "sigmoid":
            raise ValueError(
                "kl_sparsity needs latent_activation='sigmoid': the KL formula is "
                "only defined for average activations in (0, 1)."
            )
        if self.variational and self.hyperspherical:
            raise ValueError(
                "A Gaussian prior and a unit-sphere projection contradict each "
                "other. Use one or the other (see the exercises for the fix)."
            )
        if self.noise_type != "none" and self.noise_level <= 0:
            raise ValueError("noise_type is set but noise_level is 0.")

    def describe(self):
        """Human-readable list of the active terms."""
        on = []
        if self.noise_type != "none":
            on.append(f"denoising({self.noise_type}, {self.noise_level})")
        if self.variational:
            on.append(f"variational(beta={self.beta})")
        if self.hyperspherical:
            on.append("hyperspherical")
        if self.l1_latent:
            on.append(f"L1({self.l1_latent})")
        if self.kl_sparsity:
            on.append(f"KL-sparsity({self.kl_sparsity}, rho={self.sparsity_rho})")
        if self.contractive:
            on.append(f"contractive({self.contractive})")
        if self.svdd:
            on.append(f"SVDD({self.svdd})")
        if self.weight_decay:
            on.append(f"weight_decay({self.weight_decay})")
        return ", ".join(on) if on else "no regularization"


print(AEConfig().describe())
''')

code(r'''
def _activation(kind):
    return {"leaky_relu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid(),
            "none": nn.Identity()}[kind]


class ConfigurableAutoencoder(nn.Module):
    """One model, every variant.

    forward() returns a dict so the loss function can see everything it needs
    (reconstruction, latent, and the variational parameters when they exist).
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg

        # ---- encoder body: input -> ... -> last hidden ----
        dims = [cfg.input_dim, *cfg.hidden_dims]
        body = []
        for a, b in zip(dims[:-1], dims[1:]):
            body += [nn.Linear(a, b), nn.LeakyReLU()]
        self.encoder_body = nn.Sequential(*body)

        # ---- latent head ----
        last_hidden = dims[-1]
        if cfg.variational:
            # Two heads: the mean and the log-variance of q(z|x).
            self.fc_mu = nn.Linear(last_hidden, cfg.latent_dim)
            self.fc_logvar = nn.Linear(last_hidden, cfg.latent_dim)
        else:
            self.fc_z = nn.Sequential(
                nn.Linear(last_hidden, cfg.latent_dim),
                _activation(cfg.latent_activation),
            )

        # ---- decoder: mirror of the encoder ----
        rev = [cfg.latent_dim, *reversed(cfg.hidden_dims)]
        dec = []
        for a, b in zip(rev[:-1], rev[1:]):
            dec += [nn.Linear(a, b), nn.LeakyReLU()]
        dec += [nn.Linear(rev[-1], cfg.input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

        # Deep SVDD centre, filled in before training starts.
        self.register_buffer("center", torch.zeros(cfg.latent_dim))

    def encode(self, x):
        h = self.encoder_body(x)
        if self.cfg.variational:
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            # Reparameterization trick: sample, but keep the gradient path.
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std) if self.training else mu
        else:
            mu = logvar = None
            z = self.fc_z(h)

        if self.cfg.hyperspherical:
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return {"x_hat": self.decoder(z), "z": z, "mu": mu, "logvar": logvar}


print(ConfigurableAutoencoder(AEConfig()))
''')

code(r'''
def corrupt(x, cfg: AEConfig):
    """Denoising: build the corrupted input the model actually sees."""
    if cfg.noise_type == "none":
        return x
    if cfg.noise_type == "gaussian":
        # Additive noise, clamped back into the scaler's [0, 1] range.
        return (x + cfg.noise_level * torch.randn_like(x)).clamp(0.0, 1.0)
    if cfg.noise_type == "masking":
        # Randomly zero out whole features -- "reconstruct the missing field".
        mask = (torch.rand_like(x) > cfg.noise_level).float()
        return x * mask
    raise ValueError(f"unknown noise_type: {cfg.noise_type}")


def ae_loss(cfg: AEConfig, out: dict, x_clean, x_input=None, center=None):
    """Compose the loss from whichever terms the config switched on.

    Returns a dict of named components; their sum is what we backprop.
    Every component is kept separate so you can watch the terms trade off.
    """
    parts = {}

    # 1. Reconstruction -- always present. Note the target is the CLEAN x,
    #    even when the model was fed a corrupted one.
    parts["recon"] = nn.functional.mse_loss(out["x_hat"], x_clean)

    # 2. Variational: pull q(z|x) toward N(0, I).
    if cfg.variational and cfg.beta > 0:
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        # Scale per-dimension so beta stays comparable across latent sizes.
        parts["kl"] = cfg.beta * kl / cfg.latent_dim

    # 3. Sparse, L1 flavour.
    if cfg.l1_latent > 0:
        parts["l1"] = cfg.l1_latent * out["z"].abs().mean()

    # 4. Sparse, KL flavour (needs sigmoid activations, enforced by AEConfig).
    if cfg.kl_sparsity > 0:
        rho = cfg.sparsity_rho
        rho_hat = out["z"].mean(dim=0).clamp(1e-6, 1 - 1e-6)
        kl_sp = (rho * torch.log(rho / rho_hat)
                 + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).sum()
        parts["sparse_kl"] = cfg.kl_sparsity * kl_sp

    # 5. Contractive: ||J_f(x)||_F^2 via the Hutchinson estimator.
    #    E_v ||J^T v||^2 = ||J||_F^2 for v ~ N(0, I), so one backward pass
    #    replaces latent_dim of them.
    if cfg.contractive > 0:
        if x_input is None or not x_input.requires_grad:
            raise RuntimeError("contractive needs x_input.requires_grad_(True)")
        v = torch.randn_like(out["z"])
        (grad,) = torch.autograd.grad(
            (out["z"] * v).sum(), x_input, create_graph=True,
        )
        parts["contractive"] = cfg.contractive * grad.pow(2).sum(dim=1).mean()

    # 6. Deep SVDD: collapse normal data into a tight hypersphere around c.
    #    ||z - c||^2 averaged over the batch. The distance to c then doubles as
    #    an anomaly score at inference time.
    if cfg.svdd > 0:
        if center is None:
            raise RuntimeError("svdd needs the hypersphere centre")
        parts["svdd"] = cfg.svdd * (out["z"] - center).pow(2).sum(dim=1).mean()

    return parts


print("Loss components defined.")
''')

code(r'''
def init_svdd_center(model, X_t, eps=0.1):
    """Deep SVDD centre = mean latent of an untrained forward pass.

    Components too close to zero are nudged away: a centre coordinate at exactly
    0 lets the network satisfy the penalty by collapsing that unit to zero --
    the trivial solution Deep SVDD is famous for.
    """
    model.eval()
    with torch.no_grad():
        z, _, _ = model.encode(X_t.to(DEVICE))
    c = z.mean(dim=0)
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c


def train_ae(cfg: AEConfig, verbose=True):
    """Train any variant. Returns (model, history)."""
    set_seed()
    model = ConfigurableAutoencoder(cfg).to(DEVICE)

    if cfg.svdd > 0:
        model.center = init_svdd_center(model, X_fit_t)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(patience=cfg.patience)
    history = {"train_total": [], "val_total": [], "val_recon": [], "components": []}

    for epoch in range(cfg.epochs):
        # ---------------- train ----------------
        model.train()
        running, comp_sums = 0.0, {}
        for (batch,) in train_loader:
            x_clean = batch.to(DEVICE)
            x_input = corrupt(x_clean, cfg)
            if cfg.contractive > 0:
                x_input = x_input.clone().requires_grad_(True)

            out = model(x_input)
            parts = ae_loss(cfg, out, x_clean, x_input, center=model.center)
            total = sum(parts.values())

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            running += total.item()
            for k, v in parts.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v.item()

        n = len(train_loader)
        history["train_total"].append(running / n)
        history["components"].append({k: v / n for k, v in comp_sums.items()})

        # ---------------- validate ----------------
        # No corruption at validation time: we measure clean reconstruction,
        # which is the number that is comparable across variants.
        model.eval()
        val_total, val_recon = 0.0, 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                x = batch.to(DEVICE)
                out = model(x)
                val_recon += nn.functional.mse_loss(out["x_hat"], x).item()
                # contractive needs grad, so it is dropped from the val number.
                parts = ae_loss(replace(cfg, contractive=0.0), out, x, None,
                                center=model.center)
                val_total += sum(p.item() for p in parts.values())
        history["val_total"].append(val_total / len(val_loader))
        history["val_recon"].append(val_recon / len(val_loader))

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:3d}/{cfg.epochs}  "
                  f"train {history['train_total'][-1]:.6f}  "
                  f"val_recon {history['val_recon'][-1]:.6f}")

        if stopper.step(history["val_recon"][-1]):
            if verbose:
                print(f"  early stopping at epoch {epoch + 1}")
            break

    return model, history


print("Training loop ready.")
''')

# --------------------------------------------------------------------------
# 6. Gallery
# --------------------------------------------------------------------------
md(r"""
---
## 6. The variant gallery

Now the payoff of building it as switches: each variant below is **one line**.
Notice `hybrid_dvae` and `sparse_denoising` — combining terms is not a special
case, it is just setting two fields.

> Training all eleven takes about **3–4 minutes** on a Colab CPU. Trim the
> `CONFIGS` list if you are short on time during a live session.
""")

code(r'''
CONFIGS = [
    AEConfig(name="vanilla"),

    AEConfig(name="sparse_l1", l1_latent=1e-3),

    AEConfig(name="sparse_kl", latent_activation="sigmoid",
             kl_sparsity=1e-3, sparsity_rho=0.05),

    AEConfig(name="denoising", noise_type="gaussian", noise_level=0.10),

    AEConfig(name="denoising_mask", noise_type="masking", noise_level=0.15),

    AEConfig(name="vae", variational=True, beta=1e-3),

    AEConfig(name="contractive", contractive=1e-4),

    AEConfig(name="hyperspherical", hyperspherical=True),

    AEConfig(name="svdd", svdd=1e-2),

    # --- combinations: just set more than one field ---
    AEConfig(name="sparse_denoising", noise_type="gaussian", noise_level=0.10,
             l1_latent=1e-3),

    AEConfig(name="hybrid_dvae", variational=True, beta=1e-3,
             noise_type="gaussian", noise_level=0.10, l1_latent=5e-4),
]

for cfg in CONFIGS:
    print(f"{cfg.name:20s} {cfg.describe()}")
''')

code(r'''
results, models, histories = [], {}, {}

for cfg in CONFIGS:
    print(f"\n=== {cfg.name} === {cfg.describe()}")
    model, history = train_ae(cfg, verbose=True)
    models[cfg.name] = model
    histories[cfg.name] = history

    X_hat = reconstruct(model, X_test_t)
    results.append({
        "variant": cfg.name,
        "test_mse": float(((X_test - X_hat) ** 2).mean()),
        "val_recon": history["val_recon"][-1],
        "epochs": len(history["val_recon"]),
        "terms": cfg.describe(),
    })

results_df = pd.DataFrame(results).sort_values("test_mse").reset_index(drop=True)
results_df
''')

md(r"""
### Reading this table correctly

**Lower reconstruction error is *not* automatically better.** A model that
reconstructs everything perfectly has learned the identity function — it
reconstructs the pearls too, and is useless for anomaly detection. The regularized
variants *should* score slightly worse; that is them refusing to memorise.

You want a model that reconstructs the *majority* well and the *anomalies* badly.
The gap matters, not the average.
""")

code(r'''
fig, ax = plt.subplots(figsize=(8.5, 5))
order = results_df.sort_values("test_mse", ascending=False)
ypos = np.arange(len(order))
ax.barh(ypos, order["test_mse"], color=PALETTE[0], height=0.62)
ax.set_yticks(ypos)
ax.set_yticklabels(order["variant"])
for i, v in enumerate(order["test_mse"]):
    ax.text(v, i, f"  {v:.5f}", va="center", fontsize=8, color=INK_2)
ax.set_xlabel("test reconstruction MSE (scaled units)")
ax.set_title("Reconstruction error by variant — lower is not always better")
ax.xaxis.grid(True)
ax.yaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.show()
''')

code(r'''
# The VAE trade-off, as two charts rather than one chart with two y-axes:
# reconstruction and KL live on different scales, and forcing them onto a shared
# axis (or a twin axis) would misrepresent both.
vae_components = histories["vae"]["components"]
epochs = range(1, len(vae_components) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)

axes[0].plot(epochs, [c["recon"] for c in vae_components], color=PALETTE[0])
axes[0].set_ylabel("MSE")
axes[0].set_xlabel("epoch")
axes[0].set_title("VAE — reconstruction term")
tidy(axes[0])

axes[1].plot(epochs, [c.get("kl", 0.0) for c in vae_components], color=PALETTE[1])
axes[1].set_ylabel("weighted KL")
axes[1].set_xlabel("epoch")
axes[1].set_title("VAE — KL term (pull toward the prior)")
tidy(axes[1])

fig.tight_layout()
plt.show()
''')

md(r"""
### Looking inside the latent space

Five dimensions don't fit on a screen, so we project to two with PCA. Watch the
*shapes* — and the *axis ranges*:

- **vanilla** — an arbitrary cloud; only reconstruction shaped it.
- **sparse_l1** — points collapse toward the origin: L1 pushes coordinates to zero.
- **denoising** — a narrow band; corruption forces the encoder to drop noisy directions.
- **vae** — a rounded, filled cloud: the Gaussian prior at work (clearest difference here).
- **hyperspherical** — a caution: every latent has length 1, but a 2D view of a 5D
  sphere is a *filled* disc, not a ring. Don't read a projection as the space itself.
- **svdd** — check the tick labels: the whole cloud spans a fraction of the vanilla
  range. The compactness penalty crushes normality toward the centre — which is what
  makes distance-from-centre a usable score.
""")

code(r'''
@torch.no_grad()
def latents(model, X_t):
    model.eval()
    z, mu, _ = model.encode(X_t.to(DEVICE))
    # For a VAE the *mean* is the deterministic embedding worth plotting.
    return (mu if mu is not None else z).cpu().numpy()


show = ["vanilla", "sparse_l1", "denoising", "vae", "hyperspherical", "svdd"]
price = properties_test_df["price"].values / 1e6

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, name in zip(axes.ravel(), show):
    Z = latents(models[name], X_test_t)
    Z2 = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=price, cmap=SEQ_BLUE,
                    s=7, alpha=0.6, linewidths=0)
    ax.set_title(f"{name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    tidy(ax, ygrid_only=False)

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.015, fraction=0.02)
cbar.set_label("price (millions COP)", color=INK_2, fontsize=9)
cbar.outline.set_visible(False)
fig.suptitle("Latent space (PCA to 2D), coloured by price",
             x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
plt.show()
''')

# --------------------------------------------------------------------------
# 6b. Interactive: how each variant works, vs vanilla
# --------------------------------------------------------------------------
md(r"""
### Interactive: how each variant *works*, next to vanilla

The gallery above tells you *which* variant scored what. This panel is about the
*idea* instead: pick a variant from the dropdown and see, side by side with the
vanilla autoencoder, **the one mechanism that makes it different** — plus its
advantage and disadvantage for hunting good anomalies.

Each picture is drawn from the models you just trained, but the point is the
*concept*, not the score. Run it in Colab for the live dropdown; a static render
shows the default selection.
""")

code(r'''
import ipywidgets as widgets
from IPython.display import HTML, display

# One representative model per concept (all already trained above).
CONCEPTS = {
    "denoising": {
        "model": "denoising",
        "how": "Corrupts every input with noise, then asks the model to rebuild "
               "the <b>clean</b> original. It cannot simply copy the input — the "
               "input is wrong.",
        "pro": "Refuses to memorise, so a genuine bargain is less likely to be "
               "reconstructed faithfully and hidden. Robust to messy listings.",
        "con": "The noise level is a knob: too high blurs real structure, too low "
               "behaves like vanilla.",
    },
    "sparse": {
        "model": "sparse_l1",
        "how": "Adds a penalty that pushes latent activations toward zero, so only "
               "a few units fire for any given listing.",
        "pro": "A compact, more interpretable code — each active unit tends to "
               "capture one kind of structure.",
        "con": "Over-penalise and the bottleneck loses capacity; reconstruction and "
               "the anomaly signal both degrade.",
    },
    "variational": {
        "model": "vae",
        "how": "The encoder outputs a <b>distribution</b> (a mean and a spread) "
               "instead of a single point; we sample from it and pull it toward a "
               "standard normal.",
        "pro": "A smooth, continuous latent where 'how unlikely is this listing?' "
               "has a real answer — a probabilistic anomaly score.",
        "con": "One more term to balance; push it too hard and the latent collapses "
               "to the prior (posterior collapse) and stops carrying information.",
    },
    "contractive": {
        "model": "contractive",
        "how": "Penalises how far the latent moves when the input is jittered, so "
               "the encoding is locally stable.",
        "pro": "Robust features: small measurement noise does not flip the verdict, "
               "giving stable anomaly scores.",
        "con": "Silent about directions the data barely varies in; the Jacobian "
               "penalty adds compute.",
    },
    "hyperspherical": {
        "model": "hyperspherical",
        "how": "Projects every latent onto the unit sphere, so only its "
               "<b>direction</b> survives — magnitude is thrown away.",
        "pro": "Similarity becomes an angle; anomalies that differ in <i>kind</i> "
               "(direction) stand out cleanly.",
        "con": "Discards magnitude, which can itself be informative; contradicts a "
               "Gaussian prior, so it will not mix with a plain VAE.",
    },
    "svdd": {
        "model": "svdd",
        "how": "Pulls all normal latents toward a single centre, so normality "
               "collapses into a tight ball.",
        "pro": "Distance from the centre is a direct one-class anomaly score — a "
               "second, independent signal alongside reconstruction.",
        "con": "Assumes normality is one blob; if the market has several normal "
               "regimes, a single centre is too crude.",
    },
}

# A fixed sample + one example listing, reused by the illustrations.
set_seed()
_samp = np.random.choice(len(X_test_t), 500, replace=False)
_Xs = X_test_t[_samp]
_one_np = X_test[_samp[0]]
_rng = np.random.default_rng(SEED)


@torch.no_grad()
def _encode_np(model, X_t, sample=False):
    """Latent codes as numpy. sample=True lets a VAE draw from q(z|x)."""
    was_training = model.training
    model.train(sample)          # encode() samples only in training mode
    z, _, _ = model.encode(X_t.to(DEVICE))
    model.train(was_training)
    return z.cpu().numpy()


def _pca2(Z):
    return PCA(n_components=2, random_state=SEED).fit_transform(Z)


print("Concept explainer ready.")
''')

code(r'''
def _draw_concept(concept, axL, axR):
    """Draw the vanilla idea (left) and the variant's twist (right)."""
    x = np.arange(N_FEATURES)

    if concept == "denoising":
        noisy = np.clip(_one_np + 0.10 * _rng.standard_normal(N_FEATURES), 0, 1)
        axL.bar(x, _one_np, color=PALETTE[0])
        axL.set_title("Vanilla: input = target (it can copy)")
        axR.bar(x, noisy, color=MUTED, label="corrupted input (fed in)")
        axR.plot(x, _one_np, color=PALETTE[1], marker="o", ms=4, lw=0,
                 label="clean target (scored against)")
        axR.set_title("Denoising: sees corrupted, must output clean")
        axR.legend(loc="upper right", fontsize=8)
        for ax in (axL, axR):
            ax.set_xlabel("feature"); ax.set_ylabel("scaled value"); tidy(ax)

    elif concept == "sparse":
        zv = np.abs(_encode_np(models["vanilla"], _Xs)).mean(0)
        zs = np.abs(_encode_np(models["sparse_l1"], _Xs)).mean(0)
        u = np.arange(len(zv))
        top = max(zv.max(), zs.max()) * 1.15
        axL.bar(u, zv, color=PALETTE[0]); axL.set_title("Vanilla: most units active")
        axR.bar(u, zs, color=PALETTE[0]); axR.set_title("Sparse: most units near zero")
        for ax in (axL, axR):
            ax.set_xlabel("latent unit"); ax.set_ylabel("mean |activation|")
            ax.set_ylim(0, top); ax.set_xticks(u); tidy(ax)

    elif concept == "variational":
        one = X_test_t[_samp[0]:_samp[0] + 1].repeat(300, 1)
        zc_van = _encode_np(models["vanilla"], one)                 # one point
        zc_vae = _encode_np(models["vae"], one, sample=True)        # a cloud
        axL.scatter(zc_van[:, 0], zc_van[:, 1], color=PALETTE[0], s=60, zorder=3)
        axL.set_title("Vanilla: one listing -> one point")
        axR.scatter(zc_vae[:, 0], zc_vae[:, 1], color=PALETTE[0], s=14, alpha=0.5,
                    linewidths=0)
        axR.set_title("VAE: one listing -> a distribution")
        for ax in (axL, axR):
            ax.set_xlabel("latent dim 1"); ax.set_ylabel("latent dim 2")
            tidy(ax, ygrid_only=False)

    elif concept == "contractive":
        jitter = np.clip(_one_np + 0.03 * _rng.standard_normal((300, N_FEATURES)),
                         0, 1).astype(np.float32)
        jt = torch.tensor(jitter)
        zv = _encode_np(models["vanilla"], jt)[:, :2]
        zc = _encode_np(models["contractive"], jt)[:, :2]
        both = np.vstack([zv, zc])
        lim = [both[:, 0].min(), both[:, 0].max(), both[:, 1].min(), both[:, 1].max()]
        axL.scatter(zv[:, 0], zv[:, 1], color=PALETTE[0], s=12, alpha=0.5, linewidths=0)
        axL.set_title("Vanilla: jittered input -> scattered latent")
        axR.scatter(zc[:, 0], zc[:, 1], color=PALETTE[0], s=12, alpha=0.5, linewidths=0)
        axR.set_title("Contractive: same jitter -> tight latent")
        for ax in (axL, axR):
            ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
            ax.set_xlabel("latent dim 1"); ax.set_ylabel("latent dim 2")
            tidy(ax, ygrid_only=False)

    elif concept == "hyperspherical":
        Z2 = _pca2(_encode_np(models["vanilla"], _Xs))
        unit = Z2 / (np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-9)
        axL.scatter(Z2[:, 0], Z2[:, 1], color=PALETTE[0], s=10, alpha=0.5, linewidths=0)
        axL.set_title("Vanilla: latents at any radius")
        theta = np.linspace(0, 2 * np.pi, 200)
        axR.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=1)
        axR.scatter(unit[:, 0], unit[:, 1], color=PALETTE[0], s=10, alpha=0.5,
                    linewidths=0)
        axR.set_title("Hyperspherical: projected onto the unit circle")
        axR.set_aspect("equal")
        for ax in (axL, axR):
            ax.set_xlabel("dir 1"); ax.set_ylabel("dir 2"); tidy(ax, ygrid_only=False)

    elif concept == "svdd":
        Zv = _pca2(_encode_np(models["vanilla"], _Xs))
        Zs = _pca2(_encode_np(models["svdd"], _Xs))
        axL.scatter(Zv[:, 0], Zv[:, 1], color=PALETTE[0], s=10, alpha=0.5, linewidths=0)
        axL.set_title("Vanilla: no notion of a centre")
        c = Zs.mean(0)
        dist = np.linalg.norm(Zs - c, axis=1)
        sc = axR.scatter(Zs[:, 0], Zs[:, 1], c=dist, cmap=SEQ_BLUE, s=12,
                         linewidths=0)
        for r in np.percentile(dist, [50, 90]):        # two reference rings
            axR.add_patch(plt.Circle(c, r, fill=False, color=MUTED, lw=1, ls="--"))
        axR.scatter([c[0]], [c[1]], color=PALETTE[1], marker="X", s=150,
                    edgecolor="white", linewidths=1.4, zorder=5)
        axR.annotate("centre c", (c[0], c[1]), textcoords="offset points",
                     xytext=(9, 8), fontsize=9, fontweight="bold", color=PALETTE[1])
        cb = plt.gcf().colorbar(sc, ax=axR, pad=0.02, fraction=0.045)
        cb.set_label("distance from centre  (= anomaly score)", fontsize=8,
                     color=INK_2)
        cb.outline.set_visible(False)
        axR.set_title("Deep SVDD: normal data collapses toward one centre")
        axR.set_aspect("equal")
        for ax in (axL, axR):
            ax.set_xlabel("latent 1"); ax.set_ylabel("latent 2"); tidy(ax, ygrid_only=False)


def explain_variant(variant="denoising"):
    info = CONCEPTS[variant]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
    _draw_concept(variant, axL, axR)
    fig.suptitle(f"{variant}   vs   vanilla", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()

    # Concept card: mechanism + the pearl-hunting trade-off. Border only, so it
    # reads in both light and dark themes.
    display(HTML(f"""
    <div style="border:1px solid #c3c2b7; border-radius:8px; padding:12px 16px;
                max-width:900px; line-height:1.5; font-size:14px;">
      <div style="margin-bottom:6px;"><b>How it works</b> — {info['how']}</div>
      <div style="margin-bottom:6px;"><b>&#10003; Advantage for pearl-hunting</b>
           — {info['pro']}</div>
      <div><b>&#10007; Disadvantage</b> — {info['con']}</div>
    </div>
    """))


widgets.interact(
    explain_variant,
    variant=widgets.Dropdown(options=list(CONCEPTS), value="denoising",
                             description="concept:"),
)
''')

# --------------------------------------------------------------------------
# 7. Good anomalies
# --------------------------------------------------------------------------
md(r"""
---
## 7. Finding *good* anomalies — the direction trick

This is the heart of the workshop — the idea that sets this project apart from
textbook anomaly detection.

A scalar error score says *how* unusual a listing is, but not in *what way*. So we
keep the whole **signed error vector** instead of its norm:

$$e = x - \hat{x} \in \mathbb{R}^{13}$$

Each coordinate is now a readable statement about one feature:

| $e_j$ | Meaning |
|-------|---------|
| $e_j \approx 0$ | Feature $j$ is exactly what the market predicts |
| $e_j > 0$ | The real value is **higher** than the model expects |
| $e_j < 0$ | The real value is **lower** than the model expects |

So a **pearl** — cheap for what it is, and large — is a listing whose error vector
points in the direction

$$d = [\,0,\ 0,\ \underbrace{-1}_{\text{price}},\ 0,\ 0,\ \underbrace{+1}_{\text{area}},\ 0,\ \dots\,]$$

`price = -1`: costs *less* than predicted. `area = +1`: *bigger* than predicted.
Everything else: don't care.

**You are not asking "find anomalies". You are asking "find anomalies shaped like
this".** Change $d$ and you change what counts as an opportunity — that is the
whole search interface.
""")

code(r'''
# Which trained model do we hunt with? The denoising variant is a good default:
# refusing to copy its input makes it less likely to reconstruct a bargain
# faithfully and hide it from us. Change this and re-run to compare.
ANOMALY_MODEL = "denoising"

hunter = models[ANOMALY_MODEL]
X_infer_hat = reconstruct(hunter, X_infer_t)

# The signed error vector -- the object everything below operates on.
E = X_infer - X_infer_hat

print(f"Hunting with: {ANOMALY_MODEL}")
print(f"Error matrix: {E.shape}   mean |e| = {np.abs(E).mean():.4f}")

pd.DataFrame(E, columns=FEATURES).describe().T[["mean", "std", "min", "max"]]
''')

code(r'''
# Build the target direction from feature names -- clearer than a bare list of 13
# numbers, and it fails loudly on a typo.
def direction(**wishes):
    """direction(price=-1, area=1) -> the 13-dim target vector."""
    d = np.zeros(N_FEATURES, dtype=np.float32)
    for feature, value in wishes.items():
        if feature not in FEATURES:
            raise KeyError(f"{feature!r} is not a feature. Choose from {FEATURES}")
        d[FEATURES.index(feature)] = value
    return d


# "Cheaper than it should be, bigger than it should be."
d_pearl = direction(price=-1.0, area=1.0)

print(pd.Series(d_pearl, index=FEATURES).to_string())
''')

md(r"""
### Two ways to search along $d$ — and why they differ

**The repository's approach** (`src/inference.py`) finds the listings whose error
vector is closest to $d$ (nearest-neighbour). The algebra shows *why* that works:

$$\lVert e - d \rVert^2 = \lVert e \rVert^2 - 2\,e \cdot d + \lVert d \rVert^2$$

$\lVert d \rVert^2$ is constant, so minimising the distance maximises

$$e \cdot d - \tfrac{1}{2}\lVert e \rVert^2$$

— it rewards alignment with $d$ **and penalises total weirdness**. A listing that is
cheap *but also* has a bogus geolocation gets pushed down: a built-in "don't show me
data-entry errors" term.

**The projection score** keeps only the first half:

$$s = e \cdot \hat{d}, \qquad \hat{d} = d / \lVert d \rVert$$

Pure alignment, no weirdness penalty — more extreme candidates, more junk. We
compute both; neither is "correct", they just encode different risk appetites.
""")

code(r'''
N_RESULTS = 10

# --- Method A: nearest neighbours in error space (the repo's approach) ---
nn_search = NearestNeighbors(n_neighbors=N_RESULTS).fit(E)
distances, indices = nn_search.kneighbors([d_pearl])
knn_idx = indices[0]

# --- Method B: projection onto the normalised direction ---
d_hat = d_pearl / np.linalg.norm(d_pearl)
projection = E @ d_hat
proj_idx = np.argsort(-projection)[:N_RESULTS]

overlap = len(set(knn_idx) & set(proj_idx))
print(f"kNN top-{N_RESULTS} and projection top-{N_RESULTS} share "
      f"{overlap} listings.\n")

comparison = pd.DataFrame({
    "kNN rank": range(1, N_RESULTS + 1),
    "kNN idx": knn_idx,
    "kNN distance": distances[0].round(4),
    "proj idx": proj_idx,
    "proj score": projection[proj_idx].round(4),
})
comparison
''')

code(r'''
def show_candidates(idx, title):
    """Actual vs. what the model expected, in real-world units."""
    actual = properties_inference_df.iloc[idx]

    # Invert the scaler to read the reconstruction as prices and square metres.
    reconstructed = pd.DataFrame(
        scaler.inverse_transform(X_infer_hat[idx]),
        columns=FEATURES, index=actual.index,
    )

    table = pd.DataFrame({
        "stratum": actual["stratum_name"].astype(int),
        "rooms": actual["rooms"].astype(int),
        "area m²": actual["area"],
        "price (M)": (actual["price"] / 1e6).round(1),
        "expected (M)": (reconstructed["price"] / 1e6).round(1),
        "discount %": (100 * (reconstructed["price"] - actual["price"])
                       / reconstructed["price"]).round(1),
        "score": projection[idx].round(4),
    })
    print(f"\n{title}")
    return table.sort_values("discount %", ascending=False)


show_candidates(proj_idx, "Top candidates by projection score")
''')

code(r'''
show_candidates(knn_idx, "Top candidates by nearest-neighbour search (repo method)")
''')

code(r'''
# Where do the candidates sit in the market? The pearls should be BELOW the
# price/area cloud -- that is what "cheaper than predicted" means visually.
fig, ax = plt.subplots(figsize=(8, 5.4))
ax.scatter(properties_inference_df["area"],
           properties_inference_df["price"] / 1e6,
           color=MUTED, s=10, alpha=0.35, linewidths=0, label="all listings")
ax.scatter(properties_inference_df["area"].iloc[proj_idx],
           properties_inference_df["price"].iloc[proj_idx] / 1e6,
           color=PALETTE[0], s=90, linewidths=1.6, edgecolor="white",
           zorder=3, label="candidate pearls")
ax.set_xlabel("area (m²)")
ax.set_ylabel("price (millions COP)")
ax.set_title("Candidates sit below the market's price/area cloud")
ax.legend(loc="upper left")
tidy(ax, ygrid_only=False)
plt.show()
''')

# --------------------------------------------------------------------------
# 8. Clustering
# --------------------------------------------------------------------------
md(r"""
---
## 8. Clustering the error space — discovering *kinds* of anomaly

Ranking by a direction assumes you already know what a bargain looks like — you had
to *invent* $d$. What if the market contains opportunity shapes you didn't think of?

So invert the problem: **cluster the error vectors** and let the data reveal which
directions exist:

$$\text{KMeans}(\{e_i\}) \rightarrow \text{clusters } C_1, \dots, C_k$$

Each centroid is a **signature of one recurring way of being abnormal** — "priced
above the model", "unusually large for the stratum", or just data-entry noise. You
read them off the centroids instead of guessing.

Then you still use your direction — but to **pick a cluster**, not a listing:

$$C^\star = \arg\max_k \ \frac{\bar{e}_k \cdot d}{\lVert \bar{e}_k \rVert \lVert d \rVert}$$
""")

code(r'''
# How many kinds of anomaly are there? Let the silhouette score suggest a k.
candidate_k = range(2, 11)
scores = []
sample = np.random.default_rng(SEED).choice(len(E), size=min(2000, len(E)),
                                            replace=False)
for k in candidate_k:
    labels_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(E)
    scores.append(silhouette_score(E[sample], labels_k[sample]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(candidate_k), scores, color=PALETTE[0], marker="o", markersize=6)
best_k = list(candidate_k)[int(np.argmax(scores))]
ax.scatter([best_k], [max(scores)], s=140, color=PALETTE[1], zorder=3,
           edgecolor="white", linewidths=1.6)
ax.annotate(f"best k = {best_k}", (best_k, max(scores)),
            textcoords="offset points", xytext=(10, -4),
            fontsize=9, color=INK_2)
ax.set_xlabel("number of clusters (k)")
ax.set_ylabel("silhouette score")
ax.set_title("How many kinds of anomaly does the market contain?")
tidy(ax)
plt.show()
''')

code(r'''
# A silhouette peak at k=2 usually just separates "normal" from "everything else".
# For reading anomaly *types* we want a few more groups than that.
K = max(best_k, 6)

kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(E)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

cluster_sizes = (pd.Series(labels).value_counts()
                 .reindex(range(K), fill_value=0).sort_index())
print(f"k = {K}")
print(cluster_sizes.rename("listings").to_frame().T.to_string())
''')

code(r'''
# The centroid heatmap: each row is one recurring "shape" of anomaly.
# Signed values -> a DIVERGING colour scale with a neutral midpoint at zero,
# so "above expectation" and "below expectation" read as opposites.
limit = np.abs(centroids).max()

fig, ax = plt.subplots(figsize=(12, 0.62 * K + 2.2))
im = ax.imshow(centroids, cmap=DIV_BR, aspect="auto",
               norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))

ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(K))
ax.set_yticklabels([f"cluster {i}  (n={cluster_sizes[i]})" for i in range(K)],
                   fontsize=9)
ax.grid(False)

# Direct labels: the numbers matter more than the exact hue.
for i in range(K):
    for j in range(N_FEATURES):
        v = centroids[i, j]
        if abs(v) > 0.02:
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 * limit else INK)

cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.02)
cbar.set_label("mean signed error  (actual − expected)", color=INK_2, fontsize=9)
cbar.outline.set_visible(False)
ax.set_title("Anomaly signatures — blue = below expectation, red = above")
fig.tight_layout()
plt.show()
''')

code(r'''
# Which cluster best matches "cheap and big"? Rank by cosine similarity.
norms = np.linalg.norm(centroids, axis=1)
cosine = (centroids @ d_hat) / np.maximum(norms, 1e-9)

ranking = pd.DataFrame({
    "cluster": range(K),
    "size": cluster_sizes.values,
    "cosine to d": cosine.round(3),
    "mean price error": centroids[:, FEATURES.index("price")].round(4),
    "mean area error": centroids[:, FEATURES.index("area")].round(4),
}).sort_values("cosine to d", ascending=False).reset_index(drop=True)

best_cluster = int(ranking.loc[0, "cluster"])
print(f"Best-aligned cluster: {best_cluster}\n")
ranking
''')

md(r"""
### Read those cosines honestly

The winning cluster's alignment with $d$ is **small** — somewhere around 0.05–0.15,
not 0.9. That is not a bug, and it is worth saying out loud in a workshop:

- A centroid is an **average over hundreds of listings**, so the idiosyncratic part
  of each error vector cancels and only a weak shared tendency survives.
- Nothing forced a cluster to align with *your* wish. The market contains the
  anomaly types it contains; "cheap **and** big" may simply not be one of its
  dominant modes. Look at the centroid heatmap — you will usually find a cluster
  that is *cheap* and a different one that is *big*, rather than one that is both.
- Clusters are **regions**, not rankings. Use them to choose *where* to look, then
  rank inside the winner with the projection score — which is exactly what the next
  two cells do.

If you want a cluster that genuinely matches a compound wish, that is a
feature-weighting problem (exercise 4), not a clustering problem.
""")

code(r'''
# PCA of the error space, highlighting the winning cluster.
# Two visual groups only (target vs. rest) -- colouring six clusters at once
# would put pairs on screen that are not reliably distinguishable.
E2 = PCA(n_components=2, random_state=SEED).fit_transform(E)
is_best = labels == best_cluster

fig, ax = plt.subplots(figsize=(8, 5.6))
ax.scatter(E2[~is_best, 0], E2[~is_best, 1], color=MUTED, s=9, alpha=0.35,
           linewidths=0, label="other clusters")
ax.scatter(E2[is_best, 0], E2[is_best, 1], color=PALETTE[0], s=16, alpha=0.85,
           linewidths=0, label=f"cluster {best_cluster} (best aligned)")
ax.set_xlabel("PC1 of the error space")
ax.set_ylabel("PC2 of the error space")
ax.set_title("Reconstruction-error space, with the pearl cluster highlighted")
ax.legend(loc="best")
tidy(ax, ygrid_only=False)
plt.show()
''')

code(r'''
# Inside the winning cluster, rank by projection score.
cluster_members = np.where(is_best)[0]
ranked = cluster_members[np.argsort(-projection[cluster_members])][:N_RESULTS]

show_candidates(ranked, f"Best listings inside cluster {best_cluster}")
''')

# --------------------------------------------------------------------------
# 9. Sanity check
# --------------------------------------------------------------------------
md(r"""
---
## 9. Sanity check — is it really a bargain?

The model can be wrong, with an obvious failure mode: a listing that is cheap
because it is genuinely bad (terrible location, mis-typed data) also looks like a
pearl.

So we check the old-fashioned way — take a candidate, find its **genuinely
comparable** listings (same stratum, size, neighbourhood), and compare prices
directly. This is `src/find_nearest_properties.py`: deliberately *not* a neural
network, an independent second opinion.

We match on location / stratum / rooms but **exclude `price` and `area`** — matching
on price would defeat the purpose, and area is what we claim is unusual.
""")

code(r'''
def find_comparables(reference_idx, n=6):
    """Comparable listings by location / stratum / rooms -- price excluded."""
    match_on = ["stratum_name", "propertyType", "rooms", "baths", "age",
                "gmaps_geolocations_lat", "gmaps_geolocations_lng"]
    cols = [FEATURES.index(f) for f in match_on]

    # Search in scaled space so every feature contributes comparably.
    knn = NearestNeighbors(n_neighbors=n + 1).fit(X_infer[:, cols])
    _, idx = knn.kneighbors([X_infer[reference_idx, cols]])
    return [i for i in idx[0] if i != reference_idx][:n]


pearl_idx = int(ranked[0])
comps = find_comparables(pearl_idx)

pearl = properties_inference_df.iloc[pearl_idx]
comps_df = properties_inference_df.iloc[comps]

print("CANDIDATE PEARL")
print(f"  stratum {int(pearl['stratum_name'])} | {int(pearl['rooms'])} rooms | "
      f"{pearl['area']:.0f} m² | {pearl['price'] / 1e6:.0f}M COP "
      f"| {pearl['price'] / pearl['area'] / 1e6:.2f}M per m²")

print("\nCOMPARABLE LISTINGS")
comparison = pd.DataFrame({
    "stratum": comps_df["stratum_name"].astype(int),
    "rooms": comps_df["rooms"].astype(int),
    "area m²": comps_df["area"],
    "price (M)": (comps_df["price"] / 1e6).round(0),
    "M per m²": (comps_df["price"] / comps_df["area"] / 1e6).round(2),
})
comparison
''')

code(r'''
pearl_ppm = pearl["price"] / pearl["area"] / 1e6
comp_ppm = (comps_df["price"] / comps_df["area"] / 1e6)

fig, ax = plt.subplots(figsize=(8, 4.4))
labels_x = [f"comp {i + 1}" for i in range(len(comp_ppm))] + ["CANDIDATE"]
values = list(comp_ppm.values) + [pearl_ppm]
colors = [MUTED] * len(comp_ppm) + [PALETTE[0]]

bars = ax.bar(labels_x, values, color=colors, width=0.62)
ax.axhline(comp_ppm.mean(), color=PALETTE[1], linestyle="--", linewidth=1.6,
           label=f"comparables mean ({comp_ppm.mean():.2f}M/m²)")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
            ha="center", va="bottom", fontsize=8, color=INK_2)

ax.set_ylabel("price per m² (millions COP)")
ax.set_title("Candidate vs. genuinely comparable listings")
ax.legend(loc="upper right")
tidy(ax)
plt.show()

verdict = "CHEAPER" if pearl_ppm < comp_ppm.mean() else "NOT cheaper"
print(f"\nVerdict: the candidate is {verdict} than its comparables "
      f"({pearl_ppm:.2f} vs {comp_ppm.mean():.2f} M/m²).")
''')

# --------------------------------------------------------------------------
# 10. Exercises
# --------------------------------------------------------------------------
md(r"""
---
## 10. Your turn

### Warm-up

1. **Change the wish.** Rebuild `d_pearl` with `direction(...)` and re-run
   Sections 7–8. Try:
   - `direction(price=-1, garages=1)` — underpriced *with parking*;
   - `direction(administration_price=-1, stratum_name=1)` — high stratum, low
     building fee;
   - `direction(price=-1, area=1, age=-1)` — cheap, big, and new.
2. **Swap the hunter.** Set `ANOMALY_MODEL` to `"vae"`, `"sparse_l1"`, or
   `"vanilla"`. Do the same listings survive? A pearl that every variant agrees on
   is a much stronger signal than one only a single model finds.
3. **Tune the bottleneck.** `AEConfig(latent_dim=2)` and `AEConfig(latent_dim=10)`.
   At what width does the model stop compressing and start memorising — and what
   happens to the anomaly scores when it does?

### Going further

4. **Weight the features.** The direction $d$ treats a unit of price error and a
   unit of area error as equally important. They are not. Add a per-feature weight
   vector $w$ and score with $s = (w \odot e) \cdot \hat{d}$.
5. **The β-VAE sweep.** Train with `beta` in `[1e-4, 1e-3, 1e-2, 1e-1]` and plot
   reconstruction error against KL. You are tracing a rate–distortion curve; find
   the elbow, and watch for *posterior collapse* at high β (the KL hits zero and
   the latent stops carrying information).
6. **Fix the contradiction.** `AEConfig` refuses `variational=True` with
   `hyperspherical=True`, because a Gaussian prior and a unit-sphere projection
   disagree about where latents should live. The real fix is the **hyperspherical
   VAE** (von Mises–Fisher prior). Implement it.
7. **Score with SVDD.** The `svdd` variant gives a second, independent anomaly
   score: $\lVert z - c \rVert$. Rank the inference set by it and compare against
   the reconstruction-based ranking. Where they disagree is the interesting part.
8. **Density instead of distance.** Replace `KMeans` with `DBSCAN` or
   `HDBSCAN`. KMeans forces every listing into some cluster; density methods can
   label a point as noise — which for anomaly detection may be exactly right.
9. **Close the loop.** `train_pipeline.py` only deploys a model when it beats the
   previous one's reconstruction error. Given Section 6's argument that lower error
   is not automatically better, propose a better deployment gate and implement it.

---

## What to take away

1. **The bottleneck is the whole idea.** Forcing 13 features through 5 dimensions
   makes the network learn the market's structure. Anomalies are what the structure
   cannot explain.
2. **Regularization terms compose.** Sparse, denoising, variational, contractive,
   SVDD — one class, different switches. Reach for the term that matches your
   failure mode instead of reaching for a new architecture.
3. **Do not collapse the error to a scalar.** The signed 13-dimensional error
   vector is the interesting object. The norm throws away the direction, and the
   direction is the thing you actually care about.
4. **A direction turns detection into search.** "Find anomalies" is not a useful
   question. "Find anomalies shaped like *this*" is — and the shape is a vector you
   control.
5. **Clustering finds the questions you did not ask.** The centroids are a catalogue
   of the ways this market is abnormal, including the ones you would not have guessed.
6. **Always verify outside the model.** The comparables check exists because the
   autoencoder cannot tell "underpriced" from "bad in a way we did not measure".

### References

- Hinton & Salakhutdinov (2006), *Reducing the Dimensionality of Data with Neural Networks*
- Vincent et al. (2008), *Extracting and Composing Robust Features with Denoising Autoencoders*
- Kingma & Welling (2013), *Auto-Encoding Variational Bayes*
- Rifai et al. (2011), *Contractive Auto-Encoders*
- Ruff et al. (2018), *Deep One-Class Classification* (Deep SVDD)
- Davidson et al. (2018), *Hyperspherical Variational Auto-Encoders*

Repository: <https://github.com/cmatteogr/medellin_ai_autoencoder_anomaly_detection>
""")


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
def build():
    for i, cell in enumerate(CELLS):
        # nbformat >= 4.5 requires a stable per-cell id.
        cell["id"] = f"cell-{i:02d}"
        # Source is stored as a list of lines, each keeping its newline.
        cell["source"] = cell["source"].splitlines(keepends=True)

    nb = {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "workshop_autoencoder_anomaly_detection.ipynb")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    print(f"Wrote {out_path}  ({len(CELLS)} cells: {n_code} code, {n_md} markdown)")


if __name__ == "__main__":
    build()
