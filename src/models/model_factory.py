import timm
from torch import nn

def get_model(model_name, num_classes):
    model = timm.create_model(model_name, pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model