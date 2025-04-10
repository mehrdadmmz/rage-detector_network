# model_setup.py
import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DROPOUT_RATE
from torchvision.models import MobileNet_V2_Weights


def get_model(pretrained=True):
    # Load MobileNet V2 with pretrained weights if desired
    model = models.mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)

    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier to match the number of classes
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )
    return model
