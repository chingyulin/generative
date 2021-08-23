import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim: int, image_channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64 * 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=4, stride=1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2),
            nn.Tanh(),  # map to -1 to 1 pixel values
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], z.shape[1], 1, 1)  # (batch_size, z_dim, 1, 1)
        generated = self.model(z)  # (batch_size, image_channels, 28, 28)
        return generated


if __name__ == "__main__":
    import torch

    z = torch.randn(7, 10)
    gen = Generator(z.shape[1], 3)
    fake = gen(z)
    assert fake.shape == torch.Size([7, 3, 28, 28])
    assert fake.min() > -1
    assert fake.max() < 1
