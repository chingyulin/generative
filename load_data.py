from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import RandomCrop


class StanfordDogDataset(Dataset):
    def __init__(self, img_dir: str) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = glob(f"{self.img_dir}/*/**jpg")

        self.transform = transforms.Compose(
            [
                RandomCrop(400, pad_if_needed=True),
                transforms.Grayscale(),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path).to(dtype=torch.float)
        try:
            img = self.transform(img)
            img = ((img / 255) - 0.5) * 2
            return img
        except TypeError:
            print(img_path, "error.")
            return self.__getitem__(0)


dataset = StanfordDogDataset(img_dir="Images/")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


if __name__ == "__main__":
    import numpy as np
    import torchvision.transforms.functional as F
    from PIL import Image

    def show_tensor_image(image_tensor):
        image_tensor = (1 + image_tensor) / 2
        image_unflat = image_tensor.detach().cpu()
        img = image_unflat.squeeze()
        img = F.to_pil_image(img)
        img = np.array(img, dtype="uint8")
        img = Image.fromarray(img)
        img.show()

    for i in dataloader:
        print(i.shape, i.max(), i.min())
        show_tensor_image(i[9])
        assert False
