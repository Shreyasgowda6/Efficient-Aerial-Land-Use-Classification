import csv
import time
from pathlib import Path

import torch
from thop import profile

from load_eurosat_dense_checkpoint import DEVICE, checkpoint, model, test_loader

model.eval()

# Test accuracy
correct = 0
total = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

# Parameter count and raw state-dict size
parameters = sum(parameter.numel() for parameter in model.parameters())
weight_bytes = sum(
    tensor.numel() * tensor.element_size()
    for tensor in model.state_dict().values()
)

# Batch-size-one model latency; input is already on the selected device
sample_images, _ = next(iter(test_loader))
single_image = sample_images[:1].to(DEVICE)

with torch.inference_mode():
    for _ in range(20):  # warm-up
        model(single_image)

    if DEVICE.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    repetitions = 200
    with torch.inference_mode():
        for _ in range(repetitions):
            model(single_image)

    if DEVICE.type == "mps":
        torch.mps.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000 / repetitions

# MACs per image are hardware-independent; profile a CPU copy after timing MPS.
image_size = checkpoint["dataset"]["image_size"]
macs, _ = profile(
    model.cpu(),
    inputs=(torch.zeros(1, 3, image_size, image_size),),
    verbose=False,
)
operations = 2 * macs  # Assignment convention: 1 MAC = 2 operations.

print(f"Device: {DEVICE}")
print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Parameters: {parameters:,}")
print(f"Raw state-dict size: {weight_bytes / (1024 ** 2):.2f} MiB")
print(f"Batch-size-one model latency: {latency_ms:.2f} ms")
print(f"MACs per image: {macs:,.0f}")
print(f"Operations per image: {operations:,.0f}")

results_path = Path("baseline_results.csv")
fieldnames = [
    "model_variant",
    "device",
    "test_accuracy_percent",
    "correct",
    "total",
    "parameters",
    "raw_state_dict_mib",
    "batch1_latency_ms",
    "macs_per_image",
    "operations_per_image",
]
baseline_row = {
    "model_variant": "baseline",
    "device": str(DEVICE),
    "test_accuracy_percent": f"{accuracy:.2f}",
    "correct": correct,
    "total": total,
    "parameters": parameters,
    "raw_state_dict_mib": f"{weight_bytes / (1024 ** 2):.2f}",
    "batch1_latency_ms": f"{latency_ms:.2f}",
    "macs_per_image": int(macs),
    "operations_per_image": int(operations),
}

# Replace the baseline row while preserving any compressed-model rows.
other_rows = []
if results_path.exists():
    with results_path.open(newline="") as file:
        for row in csv.DictReader(file):
            variant = row.get("model_variant") or "baseline"
            if variant != "baseline":
                other_rows.append({
                    field: row.get(field, "") for field in fieldnames
                })

with results_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([baseline_row, *other_rows])

print(f"Saved baseline row to {results_path}")
