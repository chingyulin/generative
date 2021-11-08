from torch import nn


class DPNet(nn.Module):
    def __init__(
        self, image_channels: int = 1, common_channels: int = 9, final_channel: int = 3
    ):
        super().__init__()
        self.conv1 = self._build_conv1(
            in_channels=image_channels, out_channels=common_channels, kernel_size=3
        )
        self.res1 = self._build_residual_block(
            in_channels=common_channels, out_channels=common_channels, kernel_size=3
        )
        self.res2 = self._build_residual_block(
            in_channels=common_channels, out_channels=common_channels, kernel_size=3
        )
        self.res3 = self._build_residual_block(
            in_channels=common_channels, out_channels=common_channels, kernel_size=3
        )
        self.res4 = self._build_residual_block(
            in_channels=common_channels, out_channels=common_channels, kernel_size=3
        )
        self.res5 = self._build_residual_block(
            in_channels=common_channels, out_channels=common_channels, kernel_size=3
        )
        self.conv2 = self._build_conv2(
            in_channels=common_channels, out_channels=common_channels, kernel_size=3
        )
        self.upsample = self._build_upsample_block(
            in_channels=common_channels,
            upscale_factor=4,
            image_channels=image_channels,
            kernel_size=3,
            final_channel=final_channel,
        )
        self.conv3 = self._build_conv3(
            in_channels=final_channel, out_channels=1, kernel_size=1
        )
        self.out = nn.Tanh()

    def _build_conv1(self, in_channels: int, out_channels: int, kernel_size: int):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.PReLU(),
        )

    def _build_conv2(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _build_conv3(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="valid",
            ),
        )

    def _build_residual_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _build_upsample_block(
        self, in_channels, upscale_factor, image_channels, kernel_size, final_channel
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=upscale_factor ** 2 * image_channels * final_channel,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.PixelShuffle(upscale_factor=upscale_factor),
            nn.BatchNorm2d(image_channels * final_channel),
            nn.PReLU(),
        )

    def forward(self, x):
        f0 = self.conv1(x)
        x = self.res1(f0)
        f1 = f0 + x
        x = self.res2(f1)
        f2 = f1 + x
        x = self.res3(f2)
        f3 = f2 + x
        x = self.res4(f3)
        f4 = f3 + x
        x = self.res5(f4)
        x = self.conv2(x)
        x = f0 + x
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.out(x)
        x = x.clamp(-1, 1)

        return x


if __name__ == "__main__":
    model = DPNet()
    import torch

    x = torch.randn((4, 1, 12, 12))
    y = model(x)
    print(y.shape)
