import torch
import torch.nn as nn
import torch_pruning as tp
from tqdm import tqdm

from load_eurosat_dense_checkpoint import (
    DEVICE,
    checkpoint,
    model,
    train_loader,
    val_loader,
)

# Build the pruning graph on CPU, then train the pruned model on MPS.
model = model.cpu()
image_size = checkpoint["dataset"]["image_size"]
example_input = torch.zeros(1, 3, image_size, image_size)

pruner = tp.pruner.MagnitudePruner(
    model,
    example_input,
    importance=tp.importance.MagnitudeImportance(p=2),
    pruning_ratio=0.50,
    ignored_layers=[model.fc],  # Keep all 10 output classes.
    round_to=8,
)

pruner.step()
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
best_val_accuracy = -1.0
best_state = None

for epoch in range(5):
    model.train()
    total_loss = 0.0
    train_total = 0

    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/5",
        unit="batch",
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        train_total += labels.size(0)

    model.eval()
    correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * correct / val_total
    average_loss = total_loss / train_total

    print(
        f"Epoch {epoch + 1}/5 | "
        f"train loss: {average_loss:.4f} | "
        f"validation accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)
model.eval()
torch.save(model.cpu(), "pruned_resnet18_50pct.pt")

parameters = sum(parameter.numel() for parameter in model.parameters())
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Pruned parameters: {parameters:,}")
print("Saved pruned_resnet18_50pct.pt")
