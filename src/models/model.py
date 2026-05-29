import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=7, freeze_backbone=True):
    """
    EfficientNet-B0 pretrained on ImageNet with a custom classifier head.
    
    freeze_backbone=True  Phase 1: only train the classifier head
    freeze_backbone=False  Phase 2: train the entire network end-to-end
    """
    # Loading EfficientNet-B0 
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all backbone layers if Phase 1
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replacing the default classifier head
    # EfficientNet-B0's original head outputs 1000 classes (ImageNet)
    # We replace it  for 8 classes
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    return model


def get_model_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")