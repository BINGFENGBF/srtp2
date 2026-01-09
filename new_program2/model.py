import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=2, pretrained=True, dropout_p=0.5):
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_backbone(model):
    for _, param in model.named_parameters():
        param.requires_grad = True
