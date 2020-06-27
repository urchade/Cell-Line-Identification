import torch
from torch import nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                              nn.Linear(self.model.classifier.in_features, 9),
                                              nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.model(x)

    def predict(self, x, logits=False):
        if not logits:
            x = self.forward(x)
        return [torch.argmax(i).item() for i in x]
