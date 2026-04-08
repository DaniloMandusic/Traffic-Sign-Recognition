import timm
from torch import nn

def create_model(model_name, num_classes, dropout, freeze_backbone, unfreeze_last_n=1):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=dropout
    )

    if freeze_backbone:
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the layers from the end
        # If n=1, it unfreezes the head. If n=4, it unfreezes the head + last 3 blocks.
        layers = list(model.children())
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model