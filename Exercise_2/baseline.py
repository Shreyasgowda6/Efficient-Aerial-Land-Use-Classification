import time

import torch

from load_eurosat_dense_checkpoint import DEVICE, model, test_loader

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

print(f"Device: {DEVICE}")
print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Parameters: {parameters:,}")
print(f"Raw state-dict size: {weight_bytes / (1024 ** 2):.2f} MiB")
print(f"Batch-size-one model latency: {latency_ms:.2f} ms")