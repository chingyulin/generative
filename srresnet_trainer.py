import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from torchvision.transforms.transforms import Resize, GaussianBlur
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from gan.srresnet import NetG, NetD
from gan.vgg import VggFeature
from gan.hed import Hed
from gan.utils import weights_init, save_tensor_images
import os
import json
import argparse


reconstruction = nn.MSELoss()
classification = nn.BCEWithLogitsLoss()

def save_test_examples(device, visual_examples, test_dataset ):
    downsample_test = transforms.Compose(
        [
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
            Resize(test_dataset.size // 4),
        ]
    )
    save_tensor_images(visual_examples, f"./exp/{save_dir}/real/hr.png")
    lr_examples = downsample_test(visual_examples)
    save_tensor_images(lr_examples, f"./exp/{save_dir}/real/lr.png")
    return lr_examples.to(device)

def d_loss_nonsaturating(gen, disc, lr, real):

    fake = gen(lr)
    disc_fake_pred = disc(fake)
    disc_real_pred = disc(real)

    disc_fake_loss = classification(
        disc_fake_pred, torch.zeros_like(disc_fake_pred)
    )
    disc_real_loss = classification(
        disc_real_pred, torch.ones_like(disc_real_pred)
    )
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def g_loss_nonsaturating(gen, disc, lr):
    fake = gen(lr)
    disc_fake_pred = disc(fake)
    gen_loss = classification(
        disc_fake_pred, torch.ones_like(disc_fake_pred)
    )
    return gen_loss

def d_loss_wasserstein_gp(gen, disc, lr, real):

    batch_size = lr.shape[0]
    fake = gen(lr)
    disc_fake_pred = disc(fake)
    disc_real_pred = disc(real)
    alpha = torch.rand(batch_size).reshape(-1, 1, 1, 1)
    mixed = alpha * fake + (1 - alpha) * real
    disc_mixed_pred = disc(mixed)
    grad = torch.autograd.grad(
      inputs = mixed,
      outputs = pred_d_mixed,
      grad_outputs = torch.ones_like(disc_mixed_pred),
      create_graph = True,
      retain_graph = True
    )[0]
    grad = grad.view(len(grad), -1)
    grad_norm = grad.norm(2, dim=1)
    penalty = (grad_norm - 1).pow(2).mean()

    disc_loss = disc_fake_pred - disc_real_pred + 10 * penalty
    return disc_loss

def g_loss_wasserstein_gp(gen, disc, lr):
    fake = gen(lr)
    disc_fake_pred = disc(fake)
    gen_loss = -disc_fake_pred.mean()
    return gen_loss


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vgg_training = args.vgg_loss
    adversarial_training = args.adv_loss
    hed_training = args.hed_loss

    if args.dataset == "dog":
        from load_dog_data import train_dataset, visual_examples, test_dataset
        img_channel = 3
    elif args.dataset == "document":
        from load_doc_data import train_dataset, visual_examples, test_dataset
        img_channel = 3
    else:
        raise NotImplemented


    if adversarial_training:
        if args.adv_type == "nonsaturating":
            d_loss = d_loss_nonsaturating
            g_loss = g_loss_nonsaturating
        elif args.adv_type == "wasserstein":
            d_loss = d_loss_nonsaturating
            g_loss = g_loss_wasserstein_gp
        else:
            raise NotImplemented

    if vgg_training:
        vgg = VggFeature().to(device)
    if hed_training:
        hed = Hed().to(device)


    # ----------------- config -----------------
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    resume = True

    
    n_epochs = 10
    batch_size = 8
    log_dir = f"./exp/{args.save_dir}/run.log"
    log_step = 10
    display_step = 100
    save_step = 1000
    cur_step = 0
    # ----------------- config -----------------
    gen = NetG(img_channel).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = NetD(img_channel).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))



    # start from mse trained model
    # gen.load_state_dict(
    #     torch.load("exp/dog_srresnet_mse_aug/checkpoints/26000.pth")
    # )

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    downsample_train = transforms.Compose(
        [
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
            Resize(train_dataset.size // 4),
        ]
    )

    lr_examples = save_test_examples(device, visual_examples, test_dataset )

    # initialize losses
    mean_disc_loss = 0
    mean_mse_loss = 0
    mean_gen_loss = 0 # gan loss
    mean_vgg_loss = 0
    mean_hed_loss = 0
    vgg_loss = hed_loss = gen_loss =0
    total_loss = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real in tqdm(dataloader):

            cur_step += 1
            cur_batch_size = len(real)
            real = real.to(device)
            lr = downsample_train(real)

            gen_opt.zero_grad()
            fake = gen(lr)
            mse_loss = reconstruction(fake, real)
            mean_mse_loss += mse_loss.item() / display_step

            mse_loss.backward(retain_graph=True)
            mean_mse_loss += mse_loss.item() / display_step
            gen_opt.step()
           
            if vgg_training:
                gen_opt.zero_grad()
                vgg.eval()
                fake = gen(lr)
                fake_vgg = vgg(fake)
                real_vgg = vgg(real).detach()
                vgg_loss = reconstruction(fake_vgg, real_vgg)
                mean_vgg_loss += vgg_loss.item() / display_step

                vgg.zero_grad()
                vgg_loss.backward(retain_graph=True)
                gen_opt.step()

            if hed_training:
                gen_opt.zero_grad()
                hed.eval()
                fake = gen(lr)
                fake_hed = hed(fake)
                real_hed = hed(real).detach()
                hed_loss = reconstruction(fake_hed, real_hed)
                mean_hed_loss += hed_loss.item() / display_step

                hed.zero_grad()
                hed_loss.backward(retain_graph=True)
                gen_opt.step()


            
            if adversarial_training:
                gen_opt.zero_grad()
                # disc loss
                # if epoch > 1: # let generator run first
                disc_opt.zero_grad()
                disc_loss = d_loss(gen, disc, lr, real)
                mean_disc_loss += disc_loss.item() / display_step
                disc_loss.backward(retain_graph = True)
                disc_opt.step()
                # gen loss
                gen.zero_grad()
                gen_loss = g_loss(gen, disc, lr)
                mean_gen_loss += gen_loss.item() / display_step                
                gen_loss.backward(retain_graph=True)
                gen_opt.step()


            if cur_step % log_step == 0 and cur_step > 0:
                write = f"Step {cur_step}: disc loss: {mean_disc_loss}, MSE loss: {mean_mse_loss}, Gen loss: {mean_gen_loss}, VGG loss: {mean_vgg_loss}, HED loss: {mean_hed_loss}\n"

                with open(log_dir, "a") as f:
                    f.write(write)


                mean_disc_loss = 0
                mean_mse_loss = 0
                mean_gen_loss = 0 # gan loss
                mean_hed_loss = 0
                mean_vgg_loss = 0

                if cur_step % display_step == 0:
                    sr_examples = gen(lr_examples)
                    save_tensor_images(
                        sr_examples, f"./exp/{save_dir}/fake/{cur_step}.png"
                    )

                if cur_step % save_step == 0:
                    torch.save(
                        gen.state_dict(), f"./exp/{save_dir}/checkpoints/{cur_step}.pth"
                    )

    torch.save(gen.state_dict(), f"./exp/{save_dir}/checkpoints/final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--vgg_loss", action="store_true")
    parser.add_argument("--adv_loss", action="store_true")
    parser.add_argument("--hed_loss", action="store_true")

    parser.add_argument("--adv_type", type=str, default = "nope")
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()

    save_dir = args.save_dir
    os.mkdir(f"exp/{save_dir}")
    for d in ["checkpoints", "real", "fake"]:
        os.mkdir(f"exp/{save_dir}/{d}")


    train(args)