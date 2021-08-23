import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from gan.discriminator import Discriminator
from gan.generator import Generator


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def save_tensor_images(image_tensor, path, num_images=25):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (1 + image_tensor) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    img = image_grid.squeeze()
    img = F.to_pil_image(img).convert("L")
    img = np.array(img, dtype="uint8")
    img = Image.fromarray(img)
    img.save(path)


z_dim = 64
display_step = 1000
batch_size = 128
lr = 0.0002
image_channels = 1
beta_1 = 0.5
beta_2 = 0.999
device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = nn.BCEWithLogitsLoss()
gen = Generator(z_dim, image_channels).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator(image_channels).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataloader = DataLoader(
    datasets.MNIST("./data", download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)


n_epochs = 500
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        disc_opt.zero_grad()
        fake_noise = torch.randn(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        gen_opt.zero_grad()
        fake_noise_2 = torch.randn(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}"
            )

            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
    
    if epoch % 5 == 0:
        save_tensor_images(fake, f"./output/fakes/{cur_step}.png")
        # save_tensor_images(real, f"./output/real_{cur_step}.png")
        torch.save(gen.state_dict(), f"./output/checkpoints/{cur_step}.pth")
