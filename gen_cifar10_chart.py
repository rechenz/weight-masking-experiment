"""Generate CIFAR-10 comparison chart."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results_42")
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

experiments = [
    ("none", "Baseline", "#2196F3", "-"),
    ("fixed_mask_decay", "Fixed Mask", "#9C27B0", "-"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

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

ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax1.text(50.5, 45, "constraint\nreleased", fontsize=8, color="gray", va="top")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_title("CIFAR-10: Test Accuracy vs Epoch (TinyViT)")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(0, 100)

# Subplot 2: constraint rate
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
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

plt.suptitle("Weight Masking Experiment — CIFAR-10 (100 epochs)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = images_dir / "cifar10_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# Bar chart
fig2, axb = plt.subplots(figsize=(7, 5))
names = ["Baseline", "Fixed Mask"]
vals = []
for key, label in [("none", "Baseline"), ("fixed_mask_decay", "Fixed Mask")]:
    with open(results_dir / f"log_{key}.json") as f:
        d = json.load(f)
    vals.append(d["best_test_acc"])

colors = ["#2196F3", "#9C27B0"]
bars = axb.bar(names, vals, color=colors)
axb.set_ylabel("Best Test Accuracy (%)")
axb.set_title("CIFAR-10: 100 epochs, constraint=50%")
axb.set_ylim(0, 100)
for bar, v in zip(bars, vals):
    axb.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    diff = v - 81.04
    if diff > 0:
        axb.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                 f"+{diff:.1f}%", ha="center", fontsize=9, color="white")
plt.tight_layout()
fig2.savefig(images_dir / "cifar10_bars.png", dpi=150, bbox_inches="tight")
print("Saved: cifar10_bars.png")
plt.close()
