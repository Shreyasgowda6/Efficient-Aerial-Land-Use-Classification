import torch
import torch.nn as nn
from torchvision import models


# Class encoding used during training
CLASS_NAMES = {
    0: "landbird",
    1: "waterbird",
}


def load_model(checkpoint_path, device):
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.to(device)
    model.eval()

    return model