# model.py

import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DROPOUT_RATE


def get_model(pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False  # Freeze base parameters

    model.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                     nn.Linear(model.last_channel, NUM_CLASSES)
                                     )
    return model
