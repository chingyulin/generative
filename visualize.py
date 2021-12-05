import math
from load_dog_data import StanfordDogDataset
from load_doc_data import RvlCdipDataset

from torch.utils.data import DataLoader
from gan.srresnet import NetG
from torchvision.transforms.transforms import Resize, GaussianBlur, Compose
from tqdm.auto import tqdm
import torch
from torchvision.transforms.functional import InterpolationMode

from torchvision.utils import make_grid
import torchvision.transforms.functional as F


def save_tensor_images(image_tensor, path, num_images=10):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (1 + image_tensor) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=7)
    img = F.to_pil_image(image_grid)
    img.save(path)


dog = StanfordDogDataset(
    img_dir="../dogs/Images/",
    annotation_dir="../dogs/test_list.mat",
    train=False,
    size=None,
    eval=True,
)
doc = RvlCdipDataset(
    img_dir="../documents/",
    annotation_dir="../labels/",
    train=False,
    size=300,
    eval=False,
)

dog_model_paths = [
    "exp/dog_srresnet_mse_aug/checkpoints/20000.pth",
    "exp/dog_srresnet_mse_vgg_aug/checkpoints/20000.pth",
    "exp/dog_srresnet_mse_hed2/checkpoints/20000.pth",
    "exp/dog_srresnet_mse_adv/checkpoints/20000.pth",
    "exp/dog_srresnet_mse_adv_wasserstein/checkpoints/20000.pth",
]

doc_model_paths = [
    "exp/doc_mse/checkpoints/40000.pth",
    "exp/doc_mse_vgg/checkpoints/40000.pth",
    "exp/doc_mse_hed/checkpoints/40000.pth",
    "exp/doc_mse_gan/checkpoints/40000.pth",
    "exp/doc_mse_wgan/checkpoints/40000.pth"
]

all_model_paths = [dog_model_paths, doc_model_paths]
all_test_datasets = [dog, doc]
names = ["dog", "doc"]

for name, model_paths, test_dataset in zip(names, all_model_paths, all_test_datasets):

    for i in range(0, 8000, 800):
        batch_imgs = []
        test_img = test_dataset[i].unsqueeze(0)
        
        max_side = max(test_img.shape)
        if max_side > 700:
            test_img = test_img[:, : , : int(800* (test_img.shape[2] / max_side)), : int(800* (test_img.shape[3] / max_side)) ]

        # make sure the size is 4 divisible
        test_img = test_img[:, : , : 4* (test_img.shape[2] // 4), : 4* (test_img.shape[3] // 4) ]

        downgrade = Compose(
                [
                    GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
                    Resize((test_img.shape[2]//4, test_img.shape[3]//4 ))

                ]
            )

        lr = downgrade(test_img).cuda()
        bicubic = Resize(
            (4 * (test_img.shape[2] // 4), 4 * (test_img.shape[3] // 4)),
            interpolation=InterpolationMode.BICUBIC,
        )(lr).cpu()


        batch_imgs.extend([test_img, bicubic])
        gen = NetG(3).cuda().eval()

        for model_path in model_paths:
            gen.load_state_dict(torch.load(model_path))
            sr = gen(lr).detach()
            batch_imgs.append(sr.cpu())
            print(sr.shape)

        batch_imgs = torch.stack(batch_imgs).squeeze(1).clamp(-1, 1)
        print(batch_imgs.shape)
        save_tensor_images(batch_imgs, f"qualitative/{name}_{i}.png")
