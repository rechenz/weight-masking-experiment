import json, torch, os
from wme import create_model
from wme.training import evaluate
from torchvision import datasets, transforms
from torch.utils import data as data_utils

device = torch.device("cuda")
results_dir = "results_42"

# Evaluate fixed_mask best model
model = create_model("vit", num_classes=10, in_channels=1)
state = torch.load(f"{results_dir}/best_constraint_fixed_mask_decay.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.cuda()

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
test_set = datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)
test_loader = data_utils.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
criterion = torch.nn.CrossEntropyLoss()

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"FixedMask best model: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")

log = {
    "config": {"constraint_mode": "fixed_mask", "constraint_schedule": "decay"},
    "best_test_acc": test_acc,
    "final_test_acc": test_acc,
}
with open(f"{results_dir}/log_fixed_mask_decay.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print("Done")
