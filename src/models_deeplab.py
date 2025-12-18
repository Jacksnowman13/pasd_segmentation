import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

#Legacy model of DeepLabV3 with ResNet50 backbone, used in my poster session 
def load_deeplab(num_classes: int, pretrained: bool = True):
    model = deeplabv3_resnet50(pretrained=pretrained, progress=True)
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model