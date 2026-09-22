# Triage Autoencoder: Architecture, Hyperparameters & Accuracy Optimization Guide

## 1. Executive Summary & Role in Observability Pipeline

In this LLM observability platform, telemetry traces and spans pass through a multi-stage triage and clustering pipeline. The **PyTorch Autoencoder** in [`src/triage.py`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/src/triage.py#L182-L288) serves as the **unsupervised structural anomaly detector**.

While deterministic hard rules capture explicit errors (such as HTTP 5xx, exceptions, or tool crashes), the Autoencoder is designed to detect **implicit performance regressions, silent stalls, and subtle behavioral anomalies** (e.g., abnormally low generation speeds, extreme prompt-to-completion ratios, uncharacteristic execution durations) without requiring hardcoded heuristic thresholds.

```
                    ┌────────────────────────────────────────┐
                    │            Raw Span Traces             │
                    └───────────────────┬────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              [ Deterministic Rules ]       [ Feature Extraction ]
              - HTTP >= 400                 - Duration, Token counts
              - Exceptions                  - Token ratios, Tool flags
              - status == "error"                      │
                         │                             ▼
                         │                  ┌─────────────────────┐
                         │                  │ PyTorch Autoencoder │
                         │                  │ (Unsupervised MSE)  │
                         │                  └──────────┬──────────┘
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         [ Problematic Spans Subset ]
                                        │
                                        ▼
                        [ UMAP + HDBSCAN Subclustering ]
                                        │
                                        ▼
                         [ LLM Semantic Root Cause ]
```

---

## 2. Core Mechanics: How the Autoencoder Works

### The Anomaly Detection Premise
The Autoencoder operates on manifold learning:
1. **Normal spans** exhibit correlated structural behaviors (e.g., duration scales proportionally with token count; standard prompt sizes yield predictable response sizes).
2. The network learns a compressed low-dimensional bottleneck representation of normal span dynamics.
3. When presented with an anomalous span (e.g., an LLM call running for 15 seconds but returning only 2 tokens, or a prompt explosion that stalls), the Autoencoder fails to accurately reconstruct the feature vector.
4. **Reconstruction Error (Mean Squared Error - MSE)** is used as the anomaly score:
   $$\text{MSE}(x, \hat{x}) = \frac{1}{D} \sum_{i=1}^{D} (x_i - \hat{x}_i)^2$$
   Spans with $\text{MSE}(x, \hat{x}) > \text{Threshold}$ are categorized as anomalous (`-1`).

---

## 3. Parameter & Hyperparameter Reference

All key hyperparameters governing the Autoencoder are centralized in [`src/config.py`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/src/config.py#L81-L87) and consumed in [`src/triage.py`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/src/triage.py#L243-L288):

| Parameter | Location | Current Value | Technical Description & Function |
| :--- | :--- | :--- | :--- |
| `TRIAGE_AE_EPOCHS` | `src/config.py` | `150` | Total full-dataset training iterations using the Adam optimizer. |
| `TRIAGE_AE_LR` | `src/config.py` | `1e-3` (`0.001`) | Learning rate for weight updates. |
| `TRIAGE_AE_HIDDEN_DIM` | `src/config.py` | `32` | Layer width for encoder entry; expands to `hidden * 2 = 64`. |
| `TRIAGE_AE_BOTTLENECK` | `src/config.py` | `6` | Dimension of the latent compression layer ($8 \to 6$). |
| `TRIAGE_AE_THRESHOLD_PCT` | `src/config.py` | `95` | Static percentile rank defining the anomaly cutoff boundary. |
| `Loss Function` | `src/triage.py` | `nn.MSELoss` | Measures element-wise point squared difference between input and output. |
| `Output Activation` | `src/triage.py` | `nn.Sigmoid()` | Squashes reconstructed feature values to the interval $[0, 1]$. |

---

## 4. Current Architecture & Identified Bottlenecks

### Current Feature Matrix (8 Dimensions)
Extracted in `_build_feature_matrix()`:
- `0`: `duration_ms` (Linear Min-Max normalized)
- `1`: `input_tokens` (Linear Min-Max normalized)
- `2`: `output_tokens` (Linear Min-Max normalized)
- `3`: `output_token_ratio` = $\frac{\text{output\_tokens}}{\text{input\_tokens} + \text{output\_tokens} + 1.0}$
- `4`: `is_chat_span` (Binary $1.0$ or $0.0$)
- `5`: `is_tool_span` (Binary $1.0$ or $0.0$)
- `6`: `has_model` (Binary $1.0$ or $0.0$)
- `7`: `has_error_message` (Binary $1.0$ or $0.0$)

### Identified Deficiencies Impacting Accuracy

1. **Severe Distortion from Heavy-Tailed Min-Max Scaling**:
   - Telemetry latencies and token counts are **log-normally distributed**.
   - If 99% of spans have durations between $100\text{ ms}$ and $1000\text{ ms}$, but a single stalled span takes $30,000\text{ ms}$, standard Min-Max compresses all healthy spans into the range $[0.003, 0.033]$.
   - As a result, the model's loss gradients become virtually insensitive to normal variance.

2. **Training Set Contamination (No Healthy-Only Filtering)**:
   - In [`src/triage.py`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/src/triage.py#L265), the Autoencoder trains across **all** spans in `data/spans.jsonl`, including explicit errors and crashes.
   - When the network trains on broken spans, it learns how to reconstruct errors with low MSE, defeating the anomaly detection mechanism.

3. **Bottleneck Compression Ratio is Too Permissive ($8 \to 6$)**:
   - Reducing 8 dimensions down to 6 is only a **25% reduction**.
   - An over-parameterized MLP ($8 \to 32 \to 64 \to 6 \to 64 \to 32 \to 8$) can memorize an identity mapping, failing to force the model to capture true feature correlations.

4. **Loss Gradient Disparity on Mixed Types**:
   - The feature vector mixes continuous variables (columns 0–3) with binary flags (columns 4–7). MSE treats a $0.2$ deviation on a binary flag identically to a $0.2$ deviation on normalized latency, causing the optimizer to overfit to binary flag reconstruction.

5. **Static Percentile Cutoff Artifacts**:
   - Hardcoding the threshold at `P95` forces exactly 5% of all ingested spans to be classified as anomalies, regardless of whether the cluster is 100% healthy or experiencing a 40% outage.

---

## 5. Blueprint to Maximize Autoencoder Accuracy

### Strategy 1: Log-Transformation & Robust Preprocessing
Replace naive linear scaling with log-transformations:

```python
# Transform power-law variables
dur_trans = np.log1p(max(dur, 0.0))
in_t_trans = np.log1p(max(in_t, 0.0))
out_t_trans = np.log1p(max(out_t, 0.0))

# Add domain-specific interaction features:
# 1. Generation Speed (tokens / second)
speed = (out_t / ((dur / 1000.0) + 1e-4))
speed_trans = np.log1p(speed)

# 2. Time-to-First-Token / Input Latency Proxy
latency_per_in_token = np.log1p(dur / (in_t + 1.0))
```

### Strategy 2: Semi-Supervised Baseline Training
Train the network **exclusively on clean / healthy spans** (where no deterministic rule fired):

```python
# Separate healthy baseline from candidate spans
healthy_indices = [
    i for i, span in enumerate(spans)
    if not span.get("error_message") 
    and span.get("status") != "error"
    and not span.get("exception_type")
]

X_train = feat_matrix[healthy_indices]

# Fit Autoencoder exclusively on healthy distribution
model.train_on(X_train)

# Predict reconstruction error across the entire dataset
all_recon_errors = model.compute_reconstruction_error(feat_matrix)
```

### Strategy 3: Calibrated Bottleneck & Regularized Architecture
For an input dimension of $10\text{–}12$, constrain the bottleneck to $3\text{–}4$ dimensions and add `BatchNorm` + `LeakyReLU`:

```python
import torch
import torch.nn as nn

class OptimizedSpanAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 32, bottleneck: int = 3, dropout_p: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden // 2, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(hidden // 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
```

### Strategy 4: Dynamic Statistical Anomaly Thresholding
Instead of a fixed 95th percentile, calculate thresholds based on the baseline distribution of healthy validation samples:

#### Option A: Tukey’s Interquartile Range (IQR) Method (Recommended)
$$\text{Threshold} = Q_3 + 1.5 \times (Q_3 - Q_1)$$
*Where $Q_1$ and $Q_3$ are the 25th and 75th percentiles of reconstruction errors on healthy validation spans.*

#### Option B: Gaussian 3-Sigma Rule
$$\text{Threshold} = \mu_{\text{clean\_error}} + 3 \times \sigma_{\text{clean\_error}}$$
*Flags any span whose reconstruction error deviates beyond 3 standard deviations from normal behavior.*

---

## 6. Full Drop-In Code Implementation

Below is the updated implementation for [`src/triage.py`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/src/triage.py) incorporating all accuracy improvements:

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def build_advanced_feature_matrix(spans: list[dict]) -> tuple[np.ndarray, StandardScaler]:
    """
    Builds a 10-dimensional feature matrix with log-transforms and interaction ratios.
    """
    raw = []
    for s in spans:
        dur   = float(s.get("duration_ms") or 0.0)
        in_t  = float(s.get("input_tokens") or 0.0)
        out_t = float(s.get("output_tokens") or 0.0)
        op    = (s.get("operation_name") or "").lower()

        # Log transformations for power-law metrics
        log_dur   = np.log1p(max(dur, 0.0))
        log_in    = np.log1p(max(in_t, 0.0))
        log_out   = np.log1p(max(out_t, 0.0))
        
        # Interaction metrics
        out_ratio = out_t / (in_t + out_t + 1.0)
        gen_speed = np.log1p(out_t / ((dur / 1000.0) + 1e-4))
        lat_per_token = np.log1p(dur / (in_t + 1.0))

        # Categorical / Indicator flags
        is_chat = 1.0 if op in ("chat", "completion", "fim") else 0.0
        is_tool = 1.0 if s.get("tool_called") else 0.0
        has_model = 1.0 if s.get("model") else 0.0
        has_error = 1.0 if (s.get("error_message") or "").strip() else 0.0

        raw.append([
            log_dur, log_in, log_out, out_ratio, 
            gen_speed, lat_per_token,
            is_chat, is_tool, has_model, has_error
        ])
    
    mat = np.array(raw, dtype=np.float32)
    scaler = StandardScaler()
    scaled_mat = scaler.fit_transform(mat)
    return scaled_mat, scaler

def run_calibrated_autoencoder(
    spans: list[dict],
    healthy_indices: list[int] | None = None,
    epochs: int = 150,
    lr: float = 1e-3,
    bottleneck_dim: int = 3
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Trains an Autoencoder on healthy baseline spans and flags anomalies dynamically.
    """
    feat_matrix, _ = build_advanced_feature_matrix(spans)
    n_samples, n_features = feat_matrix.shape

    # Baseline training split
    if healthy_indices and len(healthy_indices) > 20:
        X_train_np = feat_matrix[healthy_indices]
    else:
        X_train_np = feat_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedSpanAutoencoder(
        in_dim=n_features, 
        hidden=32, 
        bottleneck=bottleneck_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss for outlier robustness

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    X_all = torch.tensor(feat_matrix, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X_train)
        loss = loss_fn(recon, X_train).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Evaluation on all spans
    model.eval()
    with torch.no_grad():
        all_recon = model(X_all)
        per_span_loss = loss_fn(all_recon, X_all).mean(dim=1).cpu().numpy()

    # Compute dynamic threshold using IQR on training set
    train_recon_losses = per_span_loss[healthy_indices] if healthy_indices else per_span_loss
    q25 = float(np.percentile(train_recon_losses, 25))
    q75 = float(np.percentile(train_recon_losses, 75))
    iqr = q75 - q25
    dynamic_threshold = q75 + (1.5 * iqr)

    # Classify: -1 = anomaly, 1 = normal
    labels = np.where(per_span_loss > dynamic_threshold, -1, 1).astype(int)
    return per_span_loss, labels, dynamic_threshold
```

---

## 7. Verification & Benchmarking

To verify that these optimizations improve triage accuracy:

1. **Separation Metric (Cohen's $d$)**:
   Measure the statistical separation between known-error spans and healthy spans in reconstruction error:
   $$d = \frac{\mu_{\text{error}} - \mu_{\text{healthy}}}{\sigma_{\text{pooled}}}$$
   *A higher $d$ ($> 2.0$) indicates clear separation between healthy spans and anomalous outliers.*
2. **Cluster Purity in Subsequent Stages**:
   Run [`scratch/evaluate.py`](file:///c:/Users/Dhruvkumar/Desktop/Observability_Clusters/scratch/evaluate.py) to confirm that micro-clusters formed downstream have higher semantic coherence and fewer false-positive normal spans grouped into error clusters.
