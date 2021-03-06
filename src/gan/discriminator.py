import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, image_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=0),
        )
        self.fc1 = nn.Linear(9604, 2304)
        self.leaky = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(2304, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """assumes x has shape (batch_size, image_channels, 28, 28), so the final output has 1 x 1 size"""
        x = self.model(x).reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.leaky(x)
        x = self.fc2(x)  # no need sigmoid as it includes in BCEWithLogitLoss
        return x


if __name__ == "__main__":

    x = torch.randn(7, 1, 400, 400)
    disc = Discriminator(x.shape[1])
    preds = disc(x)
    print(preds.shape)
    print(preds)
