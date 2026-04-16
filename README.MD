# Optimized CNN Trainer

A lightweight CNN model zoo and research-grade training pipeline for multi-class image classification. Built for efficient models under 2.5M parameters — ideal for edge deployment, mobile inference, and resource-constrained environments.

## Models

| Model | Params | Key Idea |
|---|---|---|
| `edgenext_xxs` | ~1.33M | Hybrid CNN-Transformer with split depth-wise transposed attention |
| `mobileone_s0` | ~1.13M | Re-parameterizable branches that fold into plain CNN at inference |
| `ghostnetv3` | ~2.12M | Cheap linear ops to generate ghost feature maps |
| `mobilenetv4` | ~0.96M | Universal inverted bottleneck with NAS-searched architecture |

All models default to **1-channel (grayscale) input** and **4-class output**. Both are configurable via arguments.

## Metrics

The trainer computes a full academic evaluation suite after each validation epoch:

| Metric | What it measures |
|---|---|
| Accuracy | Overall correctness |
| Precision (macro) | Per-class exactness, averaged equally |
| Recall (macro) | Per-class completeness, averaged equally |
| F1-Score (macro) | Harmonic mean of precision and recall |
| AUC-ROC (macro) | Threshold-free separability (one-vs-rest) |
| Cohen's Kappa (κ) | Agreement beyond chance |
| MCC | Balanced measure using all confusion matrix quadrants |

## Figures

`trainer.plot_all()` generates five publication-ready figures:

- **Training curves** — train/val loss and accuracy vs. epoch
- **Confusion matrix** — normalized heatmap with raw counts overlay
- **ROC curves** — per-class + macro-average with AUC scores
- **Per-class metrics** — grouped bar chart of precision, recall, F1
- **Metric summary** — styled table of all global metrics

## Installation

```bash
pip install git+https://github.com/yourusername/your-repo.git
```

### Dependencies

`torch` · `timm` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

## Quick Start

```python
from models import get_model, list_models
from trainer import Trainer
from torch import nn, optim

# See available models
list_models()
# ['edgenext_xxs', 'ghostnetv3', 'mobilenetv4', 'mobileone_s0']

# Build model
model = get_model("mobileone_s0")

# Setup trainer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

trainer = Trainer(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    num_classes=4,
    class_names=["Class A", "Class B", "Class C", "Class D"],
)

# Train
history = trainer.fit(train_loader, val_loader, epochs=50)

# Evaluate
trainer.summary()
trainer.plot_all(save_dir="./figures")
```

## Kaggle Usage

```python
# Cell 1
!pip install git+https://github.com/yourusername/your-repo.git -q

# Cell 2
from models import get_model
from trainer import Trainer
```

## API Reference

### `models.py`

```python
get_model(name, **kw)       # Build a model by name
list_models()                # List registered model names
```

Every model factory accepts these keyword arguments:

| Argument | Default | Description |
|---|---|---|
| `num_classes` | `4` | Number of output classes |
| `in_chans` | `1` | Input channels (1=grayscale, 3=RGB) |
| `pretrained` | `False` | Load ImageNet weights |
| `drop_rate` | varies | Classifier dropout rate |
| `drop_path_rate` | varies | Stochastic depth rate |

### `trainer.py`

```python
Trainer(
    model,                          # nn.Module
    criterion,                      # Loss function
    optimizer,                      # Optimizer
    scheduler=None,                 # LR scheduler (optional)
    device=None,                    # Auto-detects CUDA
    num_classes=4,                  # Must match model
    class_names=None,               # For figure labels
    use_amp=True,                   # Mixed precision on GPU
    early_stopping_patience=10,     # Epochs without improvement
    gradient_clip_value=1.0,        # Max gradient norm
)
```

**Training:**

| Method | Returns | Description |
|---|---|---|
| `fit(train_dl, val_dl, epochs)` | `dict` | Full training loop with logging |
| `train_one_epoch(loader)` | `float` | Single epoch, returns train loss |
| `validate(loader)` | `(float, dict)` | Val loss + all metrics |

**Visualization:**

| Method | Description |
|---|---|
| `plot_training_curves()` | Loss and accuracy over epochs |
| `plot_confusion_matrix()` | Normalized heatmap |
| `plot_roc_curves()` | Per-class ROC with AUC |
| `plot_per_class_metrics()` | Precision / Recall / F1 bars |
| `plot_metric_summary()` | Global metrics table |
| `plot_all(save_dir)` | All figures at once |

**Utilities:**

| Method | Description |
|---|---|
| `summary()` | Print metrics + classification report |
| `get_classification_report()` | sklearn report string |

## License

MIT