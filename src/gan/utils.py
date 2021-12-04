from torch import nn
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def save_tensor_images(image_tensor, path, num_images=20):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (1 + image_tensor) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    img = F.to_pil_image(image_grid)
    img.save(path)
