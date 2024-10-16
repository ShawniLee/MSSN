import torch
from torch import nn
import torch.optim as optim

class PECCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(PECCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # flatten the tensor
        return self.classifier(features)
