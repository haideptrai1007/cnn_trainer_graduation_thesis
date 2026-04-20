"""
Trainer — A research-grade training pipeline for multi-class CNN classification.

Metrics : Precision, Recall, F1-Score, AUC-ROC, Cohen's Kappa, MCC
Figures : Training curves, confusion matrix, ROC curves, per-class bar charts,
          metric summary table
Dependencies: torch, numpy, sklearn, matplotlib, seaborn
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.amp import GradScaler, autocast


# ──────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────
class Trainer:
    """
    Multi-class classification trainer with academic-level evaluation.

    Usage
    -----
    >>> trainer = Trainer(model, criterion, optimizer, scheduler, num_classes=4)
    >>> history = trainer.fit(train_loader, val_loader, epochs=50)
    >>> trainer.plot_all()                       # every figure at once
    >>> trainer.plot_confusion_matrix()           # individual figures
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        device: Optional[torch.device] = None,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        use_amp: bool = True,
        use_data_parallel: bool = False,
        early_stopping_patience: int = 10,
        gradient_clip_value: float = 1.0,
    ):
        # Core components
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # DataParallel — wrap only when explicitly enabled and 2+ GPUs exist
        self.use_data_parallel = (
            use_data_parallel
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        )
        if self.use_data_parallel:
            self.model = nn.DataParallel(self.model)

        # Classification config
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

        # Training config
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value

        # History — stores everything for metrics and plotting
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc_roc": [],
            "kappa": [],
            "mcc": [],
            "lr": [],
            "epoch_time": [],
            # Per-class metrics (list of dicts per epoch)
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = 0
        self.patience_counter = 0

        # Store last-epoch raw outputs for final figures
        self._last_labels: Optional[np.ndarray] = None
        self._last_preds: Optional[np.ndarray] = None
        self._last_probs: Optional[np.ndarray] = None

    @property
    def _raw_model(self) -> nn.Module:
        """Unwrap DataParallel to access the underlying model."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    # ──────────────────────────────────────────────────────────
    # Training loop — single epoch
    # ──────────────────────────────────────────────────────────
    def train_one_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip_value > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_value
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / max(num_batches, 1)

    # ──────────────────────────────────────────────────────────
    # Validation loop
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(
        self, loader: torch.utils.data.DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run validation. Returns (val_loss, metrics_dict).
        Also stores raw outputs for figure generation.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        all_labels = []
        all_preds = []
        all_probs = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            probs = torch.softmax(outputs.float(), dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            running_loss += loss.item()
            num_batches += 1

        val_loss = running_loss / max(num_batches, 1)

        # Concatenate
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)

        # Store for plotting
        self._last_labels = y_true
        self._last_preds = y_pred
        self._last_probs = y_prob

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred, y_prob)

        return val_loss, metrics

    # ──────────────────────────────────────────────────────────
    # Metrics computation
    # ──────────────────────────────────────────────────────────
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Compute the full academic metric suite."""

        # --- Global metrics (macro-averaged) ---
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # AUC-ROC (one-vs-rest, macro)
        try:
            auc_roc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc_roc = 0.0  # Fallback if a class is missing in batch

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)

        # --- Per-class metrics ---
        per_class_p = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        per_class_r = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        per_class_f = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "kappa": kappa,
            "mcc": mcc,
            "per_class_precision": per_class_p,
            "per_class_recall": per_class_r,
            "per_class_f1": per_class_f,
        }

    # ──────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """
        Train the model for `epochs` epochs.

        Returns
        -------
        history : dict
            Full training history for external analysis.
        """

        if verbose:
            gpu_info = ""
            if self.use_data_parallel:
                n = torch.cuda.device_count()
                gpu_info = f" | DataParallel: {n}x {torch.cuda.get_device_name(0)}"
            print(f"{'='*70}")
            print(f"  Training on {self.device} | AMP: {self.use_amp}{gpu_info}")
            print(f"  Classes: {self.class_names}")
            print(f"  Patience: {self.early_stopping_patience}")
            print(f"{'='*70}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_loss = self.train_one_epoch(train_loader)

            # --- Validate ---
            val_loss, metrics = self.validate(val_loader)

            # --- Scheduler step ---
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Record history ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["precision"].append(metrics["precision"])
            self.history["recall"].append(metrics["recall"])
            self.history["f1"].append(metrics["f1"])
            self.history["auc_roc"].append(metrics["auc_roc"])
            self.history["kappa"].append(metrics["kappa"])
            self.history["mcc"].append(metrics["mcc"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)
            self.history["per_class_precision"].append(metrics["per_class_precision"])
            self.history["per_class_recall"].append(metrics["per_class_recall"])
            self.history["per_class_f1"].append(metrics["per_class_f1"])

            # --- Best model tracking ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self._raw_model.state_dict())
                self.patience_counter = 0
                marker = " ★"
            else:
                self.patience_counter += 1
                marker = ""

            # --- Logging ---
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} │ "
                    f"Loss: {train_loss:.4f} / {val_loss:.4f} │ "
                    f"Acc: {metrics['accuracy']:.4f} │ "
                    f"F1: {metrics['f1']:.4f} │ "
                    f"κ: {metrics['kappa']:.4f} │ "
                    f"MCC: {metrics['mcc']:.4f} │ "
                    f"LR: {current_lr:.2e} │ "
                    f"{epoch_time:.1f}s{marker}"
                )

            # --- Early stopping (disabled if patience <= 0) ---
            if self.early_stopping_patience > 0 and self.patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"\n✗ Early stopping at epoch {epoch}.")
                    print(f"  Best epoch: {self.best_epoch} (val_loss={self.best_val_loss:.4f})")
                break

        # Restore best model
        if self.best_model_state is not None:
            self._raw_model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\n✓ Restored best model from epoch {self.best_epoch}.")

        return self.history

    # ──────────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _apply_style(ax: plt.Axes) -> None:
        """Apply clean publication style to an axes."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    def plot_training_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training & validation loss and accuracy curves.

        Two subplots: (left) loss, (right) accuracy — both vs. epoch.
        """
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Loss
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", linewidth=1.8)
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", linewidth=1.8)
        ax1.axvline(self.best_epoch, color="gray", linestyle=":", alpha=0.6, label=f"Best (epoch {self.best_epoch})")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="medium")
        ax1.legend(fontsize=9)
        self._apply_style(ax1)

        # Accuracy
        ax2.plot(epochs, self.history["accuracy"], label="Accuracy", linewidth=1.8, color="#2ca02c")
        ax2.axvline(self.best_epoch, color="gray", linestyle=":", alpha=0.6, label=f"Best (epoch {self.best_epoch})")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.set_title("Validation Accuracy", fontsize=12, fontweight="medium")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9)
        self._apply_style(ax2)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a normalized confusion matrix heatmap.

        Uses the raw predictions from the last validation run.
        """
        if self._last_labels is None:
            raise RuntimeError("Run validate() or fit() first.")

        cm = confusion_matrix(self._last_labels, self._last_preds)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Proportion", "shrink": 0.8},
            ax=ax,
        )

        # Overlay raw counts
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(
                    j + 0.5,
                    i + 0.75,
                    f"(n={cm[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title("Confusion Matrix (normalized)", fontsize=12, fontweight="medium")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_roc_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot per-class ROC curves + macro-average.

        One-vs-rest strategy for multi-class.
        """
        if self._last_labels is None:
            raise RuntimeError("Run validate() or fit() first.")

        fig, ax = plt.subplots(figsize=(6, 5.5))
        colors = plt.cm.Set2(np.linspace(0, 1, self.num_classes))

        # Per-class
        all_fpr, all_tpr, all_auc = {}, {}, {}
        for i in range(self.num_classes):
            binary_labels = (self._last_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, self._last_probs[:, i])
            roc_auc = auc(fpr, tpr)
            all_fpr[i] = fpr
            all_tpr[i] = tpr
            all_auc[i] = roc_auc
            ax.plot(
                fpr,
                tpr,
                color=colors[i],
                linewidth=1.6,
                label=f"{self.class_names[i]} (AUC={roc_auc:.3f})",
            )

        # Macro-average ROC
        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
        mean_tpr /= self.num_classes
        macro_auc = auc(mean_fpr, mean_tpr)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="black",
            linewidth=2.2,
            linestyle="--",
            label=f"Macro-avg (AUC={macro_auc:.3f})",
        )

        # Chance line
        ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves (one-vs-rest)", fontsize=12, fontweight="medium")
        ax.legend(fontsize=9, loc="lower right")
        self._apply_style(ax)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_per_class_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Grouped bar chart of Precision, Recall, F1 per class.

        Uses the last validation epoch's per-class results.
        """
        if not self.history["per_class_precision"]:
            raise RuntimeError("Run fit() or validate() first.")

        p = self.history["per_class_precision"][-1]
        r = self.history["per_class_recall"][-1]
        f = self.history["per_class_f1"][-1]

        x = np.arange(self.num_classes)
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - width, p, width, label="Precision", color="#4c78a8")
        ax.bar(x, r, width, label="Recall", color="#f58518")
        ax.bar(x + width, f, width, label="F1-Score", color="#54a24b")

        # Value labels
        for i in range(self.num_classes):
            ax.text(i - width, p[i] + 0.02, f"{p[i]:.2f}", ha="center", fontsize=8)
            ax.text(i, r[i] + 0.02, f"{r[i]:.2f}", ha="center", fontsize=8)
            ax.text(i + width, f[i] + 0.02, f"{f[i]:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title("Per-Class Metrics", fontsize=12, fontweight="medium")
        ax.legend(fontsize=9)
        self._apply_style(ax)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_metric_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Summary table of all global metrics (last epoch) as a styled table figure.

        Shows Accuracy, Precision, Recall, F1, AUC-ROC, Cohen's κ, MCC.
        """
        if not self.history["accuracy"]:
            raise RuntimeError("Run fit() or validate() first.")

        metric_names = [
            "Accuracy",
            "Precision (macro)",
            "Recall (macro)",
            "F1-Score (macro)",
            "AUC-ROC (macro)",
            "Cohen's Kappa (κ)",
            "MCC",
        ]
        metric_values = [
            self.history["accuracy"][-1],
            self.history["precision"][-1],
            self.history["recall"][-1],
            self.history["f1"][-1],
            self.history["auc_roc"][-1],
            self.history["kappa"][-1],
            self.history["mcc"][-1],
        ]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.axis("off")

        table_data = [[name, f"{val:.4f}"] for name, val in zip(metric_names, metric_values)]
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.6)

        # Style header row
        for j in range(2):
            table[0, j].set_facecolor("#4c78a8")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            color = "#f5f5f5" if i % 2 == 0 else "white"
            for j in range(2):
                table[i, j].set_facecolor(color)

        ax.set_title(
            "Evaluation Summary",
            fontsize=13,
            fontweight="medium",
            pad=20,
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_all(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate all publication-ready figures at once.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save all figures as PNG files.

        Returns
        -------
        figs : dict
            Mapping of figure names to Figure objects.
        """
        figs = {}

        def _path(name: str) -> Optional[str]:
            if save_dir is None:
                return None
            import os
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"{name}.png")

        figs["training_curves"] = self.plot_training_curves(_path("training_curves"))
        figs["confusion_matrix"] = self.plot_confusion_matrix(_path("confusion_matrix"))
        figs["roc_curves"] = self.plot_roc_curves(_path("roc_curves"))
        figs["per_class_metrics"] = self.plot_per_class_metrics(_path("per_class_metrics"))
        figs["metric_summary"] = self.plot_metric_summary(_path("metric_summary"))

        return figs

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────
    def save_model(self, path: str, include_history: bool = False) -> None:
        """
        Save the best model checkpoint to disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint (e.g. "best_model.pth").
        include_history : bool
            If True, also saves training history and class names alongside
            the model weights.
        """
        state = self.best_model_state if self.best_model_state is not None else self._raw_model.state_dict()

        checkpoint = {"model_state_dict": state}
        if include_history:
            checkpoint["history"] = self.history
            checkpoint["class_names"] = self.class_names
            checkpoint["best_epoch"] = self.best_epoch
            checkpoint["best_val_loss"] = self.best_val_loss

        torch.save(checkpoint, path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a model checkpoint from disk.

        Parameters
        ----------
        path : str
            Path to a checkpoint saved by :meth:`save_model`.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            self._raw_model.load_state_dict(checkpoint["model_state_dict"])
            if "history" in checkpoint:
                self.history = checkpoint["history"]
            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]
            if "best_epoch" in checkpoint:
                self.best_epoch = checkpoint["best_epoch"]
            if "best_val_loss" in checkpoint:
                self.best_val_loss = checkpoint["best_val_loss"]
        else:
            # Support loading a plain state_dict
            self._raw_model.load_state_dict(checkpoint)

        self.model.to(self.device)
        print(f"✓ Model loaded from {path}")

    def get_classification_report(self) -> str:
        """Return sklearn's classification report as a formatted string."""
        if self._last_labels is None:
            raise RuntimeError("Run validate() or fit() first.")
        return classification_report(
            self._last_labels,
            self._last_preds,
            target_names=self.class_names,
            digits=4,
        )

    def summary(self) -> None:
        """Print a compact summary of the last epoch's metrics."""
        if not self.history["accuracy"]:
            print("No training history yet.")
            return

        print(f"\n{'─'*50}")
        print(f"  Training Summary — Best Epoch: {self.best_epoch}")
        print(f"{'─'*50}")
        print(f"  Accuracy       : {self.history['accuracy'][-1]:.4f}")
        print(f"  Precision (M)  : {self.history['precision'][-1]:.4f}")
        print(f"  Recall    (M)  : {self.history['recall'][-1]:.4f}")
        print(f"  F1-Score  (M)  : {self.history['f1'][-1]:.4f}")
        print(f"  AUC-ROC   (M)  : {self.history['auc_roc'][-1]:.4f}")
        print(f"  Cohen's κ      : {self.history['kappa'][-1]:.4f}")
        print(f"  MCC            : {self.history['mcc'][-1]:.4f}")
        print(f"{'─'*50}")
        print(f"\n{self.get_classification_report()}")