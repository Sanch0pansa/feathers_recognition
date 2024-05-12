import torch
from torch import nn
import torchvision.models as models


def get_classifier(in_features: int, num_classes: int = 594) -> nn.Linear:
    return nn.Linear(in_features=in_features, out_features=num_classes)


def handle_densenet_model(model: nn.Module, weights: str, save_to: str | None = None, num_classes: int = 594):
    model.classifier = get_classifier(in_features=model.classifier.in_features, num_classes=num_classes)

    if weights:
        model.load_state_dict(torch.load(weights))

    if save_to is not None:
        torch.save(model.state_dict(), save_to)

    return model


def handle_resnet_model(model: nn.Module, weights: str, save_to: str | None = None, num_classes: int = 594):
    model.fc = get_classifier(in_features=model.fc.in_features, num_classes=num_classes)

    if weights:
        model.load_state_dict(torch.load(weights))

    if save_to is not None:
        torch.save(model.state_dict(), save_to)

    return model


def get_dense121(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.densenet121(weights='DEFAULT')
    return handle_densenet_model(model, weights, save_to, num_classes)


def get_dense161(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.densenet161(weights='DEFAULT')
    return handle_densenet_model(model, weights, save_to, num_classes)


def get_dense169(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.densenet169(weights='DEFAULT')
    return handle_densenet_model(model, weights, save_to, num_classes)


def get_dense201(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.densenet201(weights='DEFAULT')
    return handle_densenet_model(model, weights, save_to, num_classes)


def get_resnet50(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.resnet50(weights='DEFAULT')
    return handle_resnet_model(model, weights, save_to, num_classes)


def get_resnet18(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.resnet18(weights='DEFAULT')
    return handle_resnet_model(model, weights, save_to, num_classes)


def get_resnet34(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.resnet34(weights='DEFAULT')
    return handle_resnet_model(model, weights, save_to, num_classes)


def get_resnet101(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.resnet101(weights='DEFAULT')
    return handle_resnet_model(model, weights, save_to, num_classes)


def get_resnet152(weights: str | None = None, save_to: str | None = None, num_classes: int = 594):
    model = models.resnet152(weights='DEFAULT')
    return handle_resnet_model(model, weights, save_to, num_classes)