"""Generate comprehensive comparison chart for all 5 experiments."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results_42")
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Load all logs
experiments = {
    "none": "Baseline",
    "dropout_decay": "Dropout",
    "actnoise_decay": "ActNoise",
    "subspace_decay": "Subspace SVD",
    "fixed_mask_decay": "Fixed Mask",
}

colors = {
    "Baseline": "#2196F3",
    "Dropout": "#F44336",
    "ActNoise": "#4CAF50",
    "Subspace SVD": "#FF9800",
    "Fixed Mask": "#9C27B0",
}
linestyles = {
    "Baseline": "-",
    "Dropout": "--",
    "ActNoise": "-.",
    "Subspace SVD": ":",
    "Fixed Mask": "-",
}

results = {}
for key, label in experiments.items():
    fpath = results_dir / f"log_{key}.json"
    if fpath.exists():
        with open(fpath) as f:
            results[label] = json.load(f)

# Plot 1: Test & Train Accuracy + Loss
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for label, log in results.items():
    c = colors.get(label, "#666")
    ls = linestyles.get(label, "-")
    epochs = range(1, len(log.get("test_acc", [])) + 1)
    if not epochs:
        continue
    axes[0, 0].plot(epochs, log.get("test_acc", []), color=c, ls=ls, lw=2, label=label)
    axes[0, 1].plot(epochs, log.get("train_acc", []), color=c, ls=ls, lw=2, label=label)
    axes[1, 0].plot(epochs, log.get("test_loss", []), color=c, ls=ls, lw=2, label=label)
    axes[1, 1].plot(epochs, log.get("train_loss", []), color=c, ls=ls, lw=2, label=label)

# Constraint boundary
for ax in axes.flat:
    ax.axvline(x=25, color="gray", linestyle="--", alpha=0.3)

axes[0, 0].set_title("Test Accuracy (%)")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)
axes[0, 1].set_title("Train Accuracy (%)")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)
axes[1, 0].set_title("Test Loss")
axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)
axes[1, 1].set_title("Train Loss")
axes[1, 1].set_xlabel("Epoch"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

plt.suptitle("Weight Masking Experiment — Activation vs Weight Constraints", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(images_dir / "full_comparison_lines.png", dpi=150, bbox_inches="tight")
print("Saved: images/full_comparison_lines.png")

# Plot 2: Bar chart summary
fig2, ax2 = plt.subplots(figsize=(10, 5))
names = []
vals = []
for label in ["Baseline", "Dropout", "ActNoise", "Fixed Mask", "Subspace SVD"]:
    if label in results:
        names.append(label)
        vals.append(results[label]["best_test_acc"])
bars = ax2.bar(names, vals, color=[colors[n] for n in names])
ax2.set_ylabel("Best Test Accuracy (%)")
ax2.set_title("FashionMNIST + TinyViT — Constraint Mode Comparison")
ax2.set_ylim(0, 100)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    diff = v - results["Baseline"]["best_test_acc"]
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
             f"{diff:+.1f}%", ha="center", fontsize=9, color="white" if v > 70 else "black")
plt.tight_layout()
fig2.savefig(images_dir / "act_and_short_result.png", dpi=150, bbox_inches="tight")
print("Saved: images/act_and_short_result.png")

plt.close("all")
print("Done!")
