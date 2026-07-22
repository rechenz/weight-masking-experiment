import json
import os

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_42")
for fname in ["log_none.json", "log_dropout_decay.json", "log_actnoise_decay.json"]:
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        mode = d.get("config", {}).get("constraint_mode", "?")
        sched = d.get("config", {}).get("constraint_schedule", "")
        tag = f"{mode}"
        if sched:
            tag += f"_{sched}"
        best = d["best_test_acc"]
        final = d.get("final_test_acc", "?")
        if isinstance(final, float):
            final = f"{final:.2f}"
        print(f"  {tag:25s} | best={best:.2f}%  final={final}%")
    else:
        print(f"  {fname:25s} | MISSING")
