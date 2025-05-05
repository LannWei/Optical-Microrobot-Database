import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, mobilenet_v2
from transformers import ViTModel

# Simple CNN for regression
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        return self.fc(x)

# VGG16 for regression
class VGG16Regression(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1], nn.ReLU())
        self.regressor = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.regressor(x)

# ViT for regression
class ViTRegressionWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.regressor = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.regressor(cls_token)

# Generic wrapper for pretrained models with adaptive head
class PretrainedBackboneRegressor(nn.Module):
    def __init__(self, backbone, in_features):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)

# Model builder
def build_depth_regression_model(name):
    name = name.lower()
    if name == "cnn":
        return SimpleCNN()
    elif name == "vgg16":
        return VGG16Regression()
    elif name == "resnet18":
        base = models.resnet18(pretrained=True)
        in_features = base.fc.in_features
        base = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        return PretrainedBackboneRegressor(base, in_features)
    elif name == "resnet50":
        base = models.resnet50(pretrained=True)
        in_features = base.fc.in_features
        base = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        return PretrainedBackboneRegressor(base, in_features)
    elif name == "efficientnet":
        base = efficientnet_b0(pretrained=True)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        return PretrainedBackboneRegressor(base, in_features)
    elif name == "mobilenetv2":
        base = mobilenet_v2(pretrained=True)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        return PretrainedBackboneRegressor(base, in_features)
    elif name == "vit":
        return ViTRegressionWrapper()
    else:
        raise ValueError(f"Unsupported model: {name}")
