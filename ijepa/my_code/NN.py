import torch.nn as nn

class EncodeClassifyNet(nn.Module):
    def __init__(self, target_encoder):
        super().__init__()

        self.model = nn.Sequential(
            target_encoder,
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(320 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)
    

class EncodeLNClassify(nn.Module):
    def __init__(self, target_encoder):
        super().__init__()

        self.model = nn.Sequential(
            target_encoder,
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.LayerNorm(320 * 64),
            nn.Linear(320 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)
    

