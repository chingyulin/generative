import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, image_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """assumes x has shape (batch_size, image_channels, 28, 28), so the final output has 1 x 1 size"""
        return self.model(x).view(x.shape[0], -1)  # (batch_size, 1)


if __name__ == "__main__":

    x = torch.randn(7, 3, 28, 28)
    disc = Discriminator(x.shape[1])
    preds = disc(x)
    assert preds.shape == torch.Size([x.shape[0], 1])
