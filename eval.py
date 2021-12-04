import math
from load_dog_data import StanfordDogDataset, visual_examples
# from load_doc_data import RvlCdipDataset, visual_examples

from torch.utils.data import DataLoader
from gan.srresnet import NetG
from torchvision.transforms.transforms import Resize, GaussianBlur, Compose
from tqdm.auto import tqdm
import torch
from torchvision.transforms.functional import InterpolationMode

test_dataset = StanfordDogDataset(
    img_dir="../dogs/Images/",
    annotation_dir="../dogs/test_list.mat",
    train=False,
    size=None,
    eval=True,
)
# test_dataset = RvlCdipDataset(
#     img_dir="../documents/",
#     annotation_dir="../labels/",
#     train=False,
#     size=100,
#     eval=True,
#     subset=True
# )
dataloader = DataLoader(test_dataset, batch_size=1)
downsample = Compose([GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1))])

# gen = NetG(3).cuda().eval()
# gen.load_state_dict(torch.load("exp/doc_mse_hed_vgg/checkpoints/40000.pth"))

psnr = []

# for hr in tqdm(dataloader):
#     # hr = hr.cuda()

#     lr = Resize((hr.shape[2] // 4, hr.shape[3] // 4))(downsample(hr))
#     # sr = gen(lr).detach()
#     sr = Resize( (4* (hr.shape[2] // 4),  4* (hr.shape[3] // 4)) , interpolation =  InterpolationMode.BICUBIC)(lr)

#     hr = ((hr/2) + 0.5) * 255
#     sr = ((sr/2) + 0.5) * 255

#     hr = hr[:, : , : 4* (hr.shape[2] // 4), : 4* (hr.shape[3] // 4) ]
#     mean_square_error = ((  sr - hr  )**2).mean()
#     psnr.append(10 * math.log10( 255**2 /  mean_square_error))

# print(sum(psnr) / len(psnr))

lr_examples = Resize((visual_examples.shape[2] // 4, visual_examples.shape[3] // 4))(
    downsample(visual_examples)
)
sr_examples = Resize(
    (4 * (visual_examples.shape[2] // 4), 4 * (visual_examples.shape[3] // 4)),
    interpolation=InterpolationMode.BICUBIC,
)(lr_examples).clamp(-1, 1)

from torchvision.utils import make_grid
import torchvision.transforms.functional as F


def save_tensor_images(image_tensor, path, num_images=5):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (1 + image_tensor) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # img = image_grid.squeeze()
    img = F.to_pil_image(image_grid)
    # img = np.array(img, dtype="uint8")
    # img = Image.fromarray(img)
    img.save(path)


save_tensor_images(sr_examples, "bicubic_dog.png")
