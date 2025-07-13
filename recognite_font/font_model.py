import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FontClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super(FontClassifier, self).__init__()
        self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output