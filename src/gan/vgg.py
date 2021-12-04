from torch import nn
from torchvision.models import vgg11
from torchvision.transforms import transforms


class VggFeature(nn.Module):
    def __init__(self) -> None:
        super(VggFeature, self).__init__()
        vgg = vgg11(pretrained=True).eval()
        self.feature = nn.Sequential(*list(vgg.features.children())[:-1])
        self.preprocessing = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def forward(self, x):
        x = self.preprocessing(x)
        return self.feature(x)
