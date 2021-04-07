from math import log2

import torch
from torch import optim
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import Gen, Disc


def plot_to_tensorboard():
    pass


def save_checkpoint():
    pass


def load_checkpoint():
    pass


def generate_examples():
    pass


def get_loader(size, batch_size, data_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]
    )
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    return loader, len(dataset)
    


def train_fn(
    critic,
    gen,
    z_dim,
    lambda_gp,
    lambda_drift,
    loader,
    epochs,
    data_len,
    device,
    step,
    alpha,
    opt_critic,
    opt_gen,
    # tensorboard_step,
    scaler_critic,
    scaler_gen,
):
    """
    Do one epoch of training.
    """
    for _, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]
        # Train Critic: max (E[critic(real)] - E[critic(fake)])
        noise = torch.randn(batch_size, z_dim).to(device)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + lambda_gp * gp
                + lambda_drift * torch.mean(critic_real ** 2)
                # "drift loss". Avoid the critic to go to far way from zero.
            )
        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Gen: max E[critic(fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += batch_size / (data_len * epochs * 0.5)
        alpha = min(alpha, 1)


def main(start_size, data_dir, z_dim, in_channels, factors, lambda_gp, lr, betas, prog_epochs, batch_size):
    device = "cuda:9"
    critic = Disc(in_channels, factors)
    gen = Gen(z_dim, in_channels, factors)
    gen.to(device)
    critic.to(device)
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=betas)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()
    critic.train()
    gen.train()
    # tensorboard_step = 0
    step = int(log2(start_size / 4))  # The starting point
    for step, (epochs, batch_size) in enumerate(zip(prog_epochs[step:], batch_size[step:]), step):
        alpha = 1e-5  # just grab a initial value: 1 / (800k * 4 / 32) = 1e-5
        size = 4 * (2 ** step)
        loader, data_len = get_loader(size, batch_size, data_dir)

        for _ in range(epochs):
            alpha = train_fn(
                critic,
                gen,
                z_dim,
                lambda_gp,
                loader,
                # ------
                epochs,
                data_len,
                # these two calculate total step number and thus the increment of alpha each step
                # ------
                device,
                step,
                alpha,
                opt_critic,
                opt_gen,
                # tensorboard_step,
                scaler_critic,
                scaler_gen,
            )
            save_checkpoint()
            save_checkpoint()


def gradient_penalty(critic, real, fake, alpha, step, device):
    batch_size, c, h, w = real.shape
    beta = torch.rand((batch_size, 1, 1, 1), device=device).repeat(1, c, h, w)
    interpolated_image = real * beta + fake.detach() * (1 - beta)
    interpolated_image.requires_grad_(True)

    mixed_score = critic(interpolated_image, alpha, step)

    gradient = torch.autograd.grad(
        inputs=interpolated_image,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


if __name__ == "__main__":
    START_SIZE = 4
    SIZE = 256
    # ------ For optimizers
    LR = 1e-3
    BETAS = 0.0, 0.99
    # ------ For loader
    DATA_DIR = "/home/tiankang/wusuowei/data/kaggle/CelebA"
    BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16]  # For resolution [4, 8, 16, 32, 64, 128, 256]
    PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES)
    # ------ For the network
    FACTORS = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    Z_DIM = 256  # 512 in original paper
    IN_CHANNELS = 256  # 512 in original paper
    # ------ For loss
    LAMBDA_GP = 10
    LAMBDA_DRIFT = 0.001
    # in the original paper, Disc sees 800k images as alpha increase, and another 800k images when alpha = 1
    # For each image, they do 4 updates (batch repeat). So number of epoch should be different for different resolution.
    main(START_SIZE, DATA_DIR, Z_DIM, IN_CHANNELS, FACTORS, LAMBDA_GP, LR, BETAS, PROGRESSIVE_EPOCHS, BATCH_SIZES)
