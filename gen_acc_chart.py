"""Generate accuracy-vs-epoch chart for README."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results_42")
experiments = [
    ("none", "Baseline", "#2196F3", "-"),
    ("dropout_decay", "Dropout", "#F44336", "--"),
    ("actnoise_decay", "ActNoise", "#4CAF50", "-."),
    ("subspace_decay", "Subspace SVD", "#FF9800", ":"),
    ("fixed_mask_decay", "Fixed Mask", "#9C27B0", "-"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for key, label, color, ls in experiments:
    fpath = results_dir / f"log_{key}.json"
    if not fpath.exists():
        continue
    with open(fpath) as f:
        log = json.load(f)
    if "test_acc" not in log or not log["test_acc"]:
        continue
    epochs = range(1, len(log["test_acc"]) + 1)
    ax1.plot(epochs, log["test_acc"], color=color, ls=ls, lw=2, label=label)

ax1.axvline(x=25, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax1.text(25.5, 98, "constraint\nreleased", fontsize=8, color="gray", va="top")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_title("Test Accuracy vs Epoch")
ax1.legend(fontsize=9, loc="lower right")
ax1.grid(True, alpha=0.2)
ax1.set_ylim(0, 100)

# Subplot 2: constraint rate per epoch
for key, label, color, ls in experiments:
    fpath = results_dir / f"log_{key}.json"
    if not fpath.exists():
        continue
    with open(fpath) as f:
        log = json.load(f)
    if "constraint_rate" not in log or not log["constraint_rate"]:
        continue
    epochs = range(1, len(log["constraint_rate"]) + 1)
    ax2.plot(epochs, log["constraint_rate"], color=color, ls=ls, lw=2, label=label)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Constraint Rate")
ax2.set_title("Constraint Rate vs Epoch (decay schedule)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

plt.suptitle("Weight Masking Experiment — Activation & Weight Constraints", fontsize=13, fontweight="bold")
plt.tight_layout()
out = Path("images/accuracy_vs_epoch.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
