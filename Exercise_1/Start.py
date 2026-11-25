import torch

from waterbirds_model import (
    load_model,
    CLASS_NAMES,
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = load_model(
    "waterbirds_erm_resnet18_biased.pt",
    device,
)

print(model)