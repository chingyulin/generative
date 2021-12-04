from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import random
from torchvision.transforms import (
    RandomCrop,
    CenterCrop,
    RandomRotation,
    RandomHorizontalFlip,
)
from scipy.io import loadmat
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

class RvlCdipDataset(Dataset):
    def __init__(
        self, img_dir: str, annotation_dir: str, size: int = 96, train=False, eval=False, subset=False
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        
        data_cat = "train" if train else "test"
        label_path = f"{annotation_dir}/{data_cat}.txt"
        with open(label_path) as f:
            self.img_paths = [l.rstrip().split(" ")[0] for l in f.readlines()]
        if subset:
            random.seed(42)
            self.img_paths = random.choices(self.img_paths, k = 4000)
        self.size = size

        if not eval:
            if train:

                t = [
                    RandomHorizontalFlip(),
                    RandomCrop(self.size, pad_if_needed=True),
                    RandomRotation(degrees=30),
                    # transforms.Grayscale(),
                ]
            else:
                t = [
                    CenterCrop(self.size),
                    # transforms.Grayscale()
                ]

            self.transform = transforms.Compose(t)
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_paths[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            img = pil_to_tensor(img)
            # n02105855-Shetland_sheepdog/n02105855_2933.jpg
            if self.transform:
                img = self.transform(img)
            img = ((img / 255) - 0.5) * 2
            return img
        except TypeError:
            print(img_path, "error.")
            return self.__getitem__(0)


train_dataset = RvlCdipDataset(
    img_dir="../documents/", annotation_dir="../labels/", train=True
)
test_dataset = RvlCdipDataset(
    img_dir="../documents/",
    annotation_dir="../labels/",
    train=False,
    size=400,
)

idxs = [i for i in range(1, len(test_dataset), 5000)]
visual_examples = torch.stack([test_dataset[idx] for idx in idxs])

if __name__ == "__main__":
    import numpy as np
    import torchvision.transforms.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader

    dataloader = DataLoader(test_dataset, batch_size=10)

    def show_tensor_image(image_tensor, i):
        image_tensor = (1 + image_tensor) / 2
        image_unflat = image_tensor.detach().cpu()
        img = image_unflat.squeeze()
        img = F.to_pil_image(img)
        img = np.array(img, dtype="uint8")
        img = Image.fromarray(img)
        img.save(f"out{i}.png")

    print(visual_examples.shape)
    for i in range(10):
        show_tensor_image(visual_examples[i], i )
        # assert False
    print(test_dataset.__len__(), train_dataset.__len__())