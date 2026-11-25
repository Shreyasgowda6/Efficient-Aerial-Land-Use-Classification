"""Student starter: load the provided dense EuroSAT ResNet-18 checkpoint.

Expected file in the working directory:
  eurosat_resnet18_dense_student_checkpoint.pt

This reconstructs:
  model, train_loader, val_loader, test_loader, classes, checkpoint

EuroSAT itself is downloaded by torchvision if it is not already present.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

CHECKPOINT_PATH = Path("eurosat_resnet18_dense_student_checkpoint.pt")
DATA_ROOT = Path("./data")
BATCH_SIZE = 64
NUM_WORKERS = 0 #2
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}\n"
        "Place the provided .pt file next to this script or change CHECKPOINT_PATH."
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

if checkpoint.get("format_version") != 1:
    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")

model_info = checkpoint["model"]
data_info = checkpoint["dataset"]
split = checkpoint["split"]

if model_info["architecture"] != "resnet18":
    raise RuntimeError(f"Unexpected architecture: {model_info['architecture']}")

# Rebuild exactly the same dense architecture. No ImageNet download is needed.
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, model_info["num_classes"])
model.load_state_dict(model_info["state_dict"], strict=True)
model = model.to(DEVICE)
model.eval()

mean = data_info["normalization"]["mean"]
std = data_info["normalization"]["std"]
image_size = data_info["image_size"]

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Two dataset instances are intentional: augmentation for recovery training, deterministic preprocessing for validation/test.
train_full = datasets.EuroSAT(root=DATA_ROOT, download=True, transform=train_transform)
eval_full = datasets.EuroSAT(root=DATA_ROOT, download=False, transform=eval_transform)

# Dataset identity checks protect against silently applying stored indices to a different class order or dataset version.
classes = list(data_info["classes"])
if list(train_full.classes) != classes:
    raise RuntimeError(
        "EuroSAT class order does not match the checkpoint.\n"
        f"Checkpoint: {classes}\nDataset: {list(train_full.classes)}"
    )

all_indices = split["train_indices"] + split["val_indices"] + split["test_indices"]
if max(all_indices) >= len(train_full):
    raise RuntimeError("Stored split indices do not fit the downloaded EuroSAT dataset.")

train_dataset = Subset(train_full, split["train_indices"])
val_dataset = Subset(eval_full, split["val_indices"])
test_dataset = Subset(eval_full, split["test_indices"])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

# Quick architecture sanity check.
with torch.inference_mode():
    dummy = torch.zeros(1, 3, image_size, image_size, device=DEVICE)
    output = model(dummy)
assert output.shape == (1, model_info["num_classes"])

print("Dense EuroSAT model loaded successfully.")
print("Device:", DEVICE)
print("Classes:", classes)
print("Split sizes:", {
    "train": len(train_dataset),
    "validation": len(val_dataset),
    "test": len(test_dataset),
})