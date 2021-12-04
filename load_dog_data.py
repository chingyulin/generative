from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import (
    RandomCrop,
    CenterCrop,
    RandomRotation,
    RandomHorizontalFlip,
)
from scipy.io import loadmat


class StanfordDogDataset(Dataset):
    def __init__(
        self, img_dir: str, annotation_dir: str, size: int = 96, train=False, eval=False
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = [f[0] for f in loadmat(annotation_dir)["file_list"][:, 0]]
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
        img_path = self.img_paths[idx]
        img = read_image(self.img_dir + img_path, ImageReadMode.RGB).to(
            dtype=torch.float
        )
        try:
            # n02105855-Shetland_sheepdog/n02105855_2933.jpg
            if self.transform:
                img = self.transform(img)
            img = ((img / 255) - 0.5) * 2
            return img
        except TypeError:
            print(img_path, "error.")
            return self.__getitem__(0)


train_dataset = StanfordDogDataset(
    img_dir="../dogs/Images/", annotation_dir="../dogs/train_list.mat", train=True
)
test_dataset = StanfordDogDataset(
    img_dir="../dogs/Images/",
    annotation_dir="../dogs/test_list.mat",
    train=False,
    size=400,
)

idxs = [i for i in range(1, len(test_dataset), 900)]
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

    # print(visual_examples.shape)
    # for i in range(10):
    #     show_tensor_image(visual_examples[i], i )
    #     # assert False
    print(test_dataset.__len__(), train_dataset.__len__())
