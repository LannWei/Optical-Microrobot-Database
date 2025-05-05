import torch.nn as nn
# import torchvision.models as models
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, mobilenet_v2
from transformers import ViTModel, ViTConfig

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_p_classes, num_r_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256), nn.ReLU()
        )
        self.head_P = nn.Linear(256, num_p_classes)
        self.head_R = nn.Linear(256, num_r_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.fc(x)
        return self.head_P(x), self.head_R(x)

# VGG with dual heads
class VGG(nn.Module):
    def __init__(self, num_p_classes, num_r_classes):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            *list(vgg.classifier.children())[:-1],  # remove last fc
            nn.ReLU()
        )
        self.head_P = nn.Linear(4096, num_p_classes)
        self.head_R = nn.Linear(4096, num_r_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.head_P(x), self.head_R(x)


# Vision Transformer with dual heads
class VisionTransformer(nn.Module):
    def __init__(self, num_p_classes, num_r_classes):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        hidden_size = self.backbone.config.hidden_size
        self.head_P = nn.Linear(hidden_size, num_p_classes)
        self.head_R = nn.Linear(hidden_size, num_r_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        return self.head_P(cls_token), self.head_R(cls_token)

# Wrapper for pretrained backbones
class MultiHeadWrapper(nn.Module):
    def __init__(self, backbone, in_features, num_p, num_r):
        super().__init__()
        self.backbone = backbone
        self.head_P = nn.Linear(in_features, num_p)
        self.head_R = nn.Linear(in_features, num_r)

    def forward(self, x):
        features = self.backbone(x)
        return self.head_P(features), self.head_R(features)


# Load backbone and wrap with two heads
def build_backbone_model(name, num_p, num_r, pretrained=True):
    name = name.lower()
    if name == "vgg16":
        return VGG(num_p, num_r)
    elif name == "resnet18":
        base = models.resnet18(pretrained=pretrained)
        in_features = base.fc.in_features
        backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    elif name == "resnet50":
        base = models.resnet50(pretrained=pretrained)
        in_features = base.fc.in_features
        backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    elif name == "efficientnet":
        base = efficientnet_b0(pretrained=pretrained)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        backbone = base
    elif name == "mobilenetv2":
        base = mobilenet_v2(pretrained=pretrained)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        backbone = base
    elif name == "vit":
        return VisionTransformer(num_p, num_r)
    else:
        raise ValueError(f"Unsupported model: {name}")

    return MultiHeadWrapper(backbone, in_features, num_p, num_r)