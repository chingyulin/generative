import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.transforms import Resize
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from gan.conditional_generator import DPNet
from gan.discriminator import Discriminator
from gan.utils import weights_init
from load_data import dataloader


def save_tensor_images(image_tensor, path, num_images=10):
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


batch_size = 8
lr = 0.0002
image_channels = 1
beta_1 = 0.5
beta_2 = 0.999
device = "cuda" if torch.cuda.is_available() else "cpu"

reconstruction = nn.MSELoss()
classification = nn.BCEWithLogitsLoss()

# gen = Generator(z_dim, image_channels).to(device)
gen = DPNet().to(device)
pth = torch.load("dog_output/checkpoints/2500.pth")
gen.load_state_dict(pth)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator(image_channels).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
# gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


display_step = 500
n_epochs = 500
cur_step = 2501
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real in tqdm(dataloader):
        cur_step += 1
        cur_batch_size = len(real)
        real = real.to(device)

        lr = Resize(100)(real)

        gen_opt.zero_grad()
        fake_2 = gen(lr)

        disc_opt.zero_grad()
        fake = gen(lr)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = classification(
            disc_fake_pred, torch.zeros_like(disc_fake_pred)
        )
        disc_real_pred = disc(real)
        disc_real_loss = classification(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        disc_fake_pred = disc(fake_2)
        gen_loss = 0.001 * classification(
            disc_fake_pred, torch.ones_like(disc_fake_pred)
        ) + reconstruction(fake_2, real)
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, ",
                f"discriminator loss: {mean_discriminator_loss}",
                f"Min: {fake_2.min()}, Max: {fake_2.max()}",
            )
            mean_generator_loss = 0
            mean_discriminator_loss = 0

            save_dir = "dog_output"
            save_tensor_images(fake_2, f"./{save_dir}/fakes/{cur_step}.png")
            save_tensor_images(lr, f"./{save_dir}/lr/{cur_step}.png")

            torch.save(gen.state_dict(), f"./{save_dir}/checkpoints/{cur_step}.pth")
