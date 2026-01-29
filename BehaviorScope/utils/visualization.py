import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None

def render_confusion_heatmap(
    matrix: List[List[float | int]],
    class_names: List[str],
    out_path: Path,
    title: str,
    normalized: bool,
    logger: logging.Logger,
) -> None:
    """
    Renders a confusion matrix heatmap using Matplotlib.
    """
    if plt is None or np is None:
        logger.warning("Matplotlib or NumPy not found. Skipping confusion plot.")
        return

    if not class_names:
        return

    matrix_np = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_np, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    if normalized:
        cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom")
    else:
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)

    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Loop over data dimensions and create text annotations.
    thresh = matrix_np.max() / 2.
    fmt = '.2f' if normalized else 'd'
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = matrix_np[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    fig.tight_layout()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        logger.error(f"Failed to save confusion heatmap: {e}")
    finally:
        plt.close(fig)

def save_training_curves(run_dir: Path, history: List[Dict], logger: logging.Logger) -> None:
    """
    Saves training curves (Loss and Accuracy) using Matplotlib.
    """
    if plt is None:
        logger.warning("Matplotlib not found. Skipping training curves.")
        return

    epochs = []
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for row in history:
        epoch = row.get("epoch")
        if epoch is None: continue
        
        train = row.get("train", {})
        val = row.get("val", {})
        
        epochs.append(epoch)
        train_loss.append(train.get("loss"))
        train_acc.append(train.get("accuracy"))
        val_loss.append(val.get("loss"))
        val_acc.append(val.get("accuracy"))

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss Plot
    ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
    ax1.plot(epochs, val_loss, label='Val Loss', marker='o')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy Plot
    ax2.plot(epochs, train_acc, label='Train Acc', marker='o')
    ax2.plot(epochs, val_acc, label='Val Acc', marker='o')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    out_path = run_dir / "training_curves.png"
    
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        logger.error(f"Failed to save training curves: {e}")
    finally:
        plt.close(fig)