import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from torchvision import datasets, transforms
from src.transforms import make_transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc1 = nn.Linear(128 * 640, 10)

    def forward(self, x):
        x = flatten(x, 1) 
        return self.fc1(x)
    


class EncodeClassifyNet(nn.Module):
    def __init__(self, target_encoder, transform):
        super().__init__()

        self.transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transform
            ])

        self.model = nn.Sequential(
            target_encoder,
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(128 * 640, 10)
        )


    def forward(self, x):
        x = self.transformation(x)
        return self.model(x)