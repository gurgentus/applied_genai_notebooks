import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    return DataLoader, F, nn, np, optim, plt, torch, torchvision, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 8: Practical 2 - Diffusion Methods""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Recall that the idea behind Diffusion Models is to map training data to a simple normal distribution by adding noise through a series of steps and then have a neural network **learn** the inverse process.

        In contrast to autoencoders, the first step is done through simple well defined functions and only the decoder is learned.
        """
    )
    return


@app.cell
def _(torch):
    # Check which GPU is available
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    return (device,)


@app.cell
def _(DataLoader, test_dataset, train_dataset):
    BATCH_SIZE = 32
    IMAGE_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return BATCH_SIZE, IMAGE_SIZE, test_loader, train_loader


@app.cell
def _(torchvision, transforms):
    # Prepare the Data
    transform = transforms.Compose([transforms.Pad(2, -1), transforms.ToTensor(),     transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return test_dataset, train_dataset, transform


@app.cell
def _(train_dataset):
    first_image, first_label = train_dataset[0]
    print(first_image.shape)
    return first_image, first_label


@app.cell
def _(first_image, first_label, plt):
    def show_image(image, label=None):
        print("Label: ", label)
        plt.imshow(image.permute(1,2,0).squeeze(), cmap='gray')
        plt.show()
    show_image(first_image, first_label)
    return (show_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's take $x_0$ to be a random variable representing images in our data distribution. We will assume that it has mean $0$ and variance $1$. This is ok since we have preprocessed the training images to have mean $0$ and variance $1$. 

        Next, we will corrupt the images by adding standard gaussian noise $\epsilon$ (mean 0, variance 1). How much noise should be added?

        To keep the transformed distribution having the same mean and variance, we can control the noise amount using a parameter $\beta$:

        $x_1 = \sqrt{1-\beta}x_0 + \sqrt{\beta} \epsilon$

        Then if $x_0$ has mean $0$ and variance $1$, $x_1$ will have mean: 

        $\sqrt{1-\beta}*0+\sqrt{\beta}*0=0$ 

        and variance: 

        $(\sqrt{1-\beta})^2*1 + (\sqrt{\beta})^2*1=1-\beta+\beta=1$.
        """
    )
    return


@app.cell
def _(first_image, np, show_image):
    def corrupt_image(image, beta = 0.1):
        image_with_noise = np.sqrt(1-beta)*image + np.sqrt(beta)*np.random.normal(0, 1, image.shape)
        return image_with_noise
    show_image(corrupt_image(first_image))
    return (corrupt_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""How does changing $\beta$ influence the amount of noise added to the image?  Try to find a value that makes the image look like a random noise image.  You can use the slider below to change the value of $\beta$.""")
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=1, step=0.05)
    slider
    return (slider,)


@app.cell
def _(corrupt_image, first_image, show_image, slider):
    show_image(corrupt_image(first_image, beta=slider.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can repeat this process using the new $x_1$ as the input to the next step, and so on, until we reach $x_T$. Moreover, we can also use a different $\beta$ for each step. The setup up of how this parameter should vary is referred to as the **diffusion schedule**.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We'll want to repeat this process through several iterations, at each step corrupting the image by a small amount. To make things simpler, we can perform some mathematical magic using the fact that multiplying a Gaussian distribution by a constant results in another Gaussian distribution. Similarly, adding two Gaussians results in a Gaussian.  So, one can show mathematically that for two standard Gaussian random variables: 

        $A \epsilon_0 + B \epsilon_1 = \left(\sqrt{A^2 + B^2}\right)\epsilon$, where $\epsilon$ is also a Gaussian random variable.

        Hence,

        $x_2 = \sqrt{1-\beta_1}x_1 + \sqrt{\beta_1} \epsilon_1 = \sqrt{1-\beta_1}(\sqrt{1-\beta_0}x_0 + \sqrt{\beta_0} \epsilon_0) + \sqrt{\beta_1} \epsilon_1 = \sqrt{(1-\beta_0)(1-\beta_1)}x_0 + \sqrt{\beta_0(1-\beta_1)}\epsilon_0 + \sqrt{\beta_1}\epsilon_1 = \sqrt{(1-\beta_0)(1-\beta_1)}x_0 + \sqrt{1 - (1-\beta_0)(1-\beta_1)}\epsilon$

        This will be used later.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's look at an example of a diffusion schedule that we'll use in our final model:""")
    return


@app.cell
def _(torch):
    def linear_diffusion_schedule(diffusion_times):
        min_rate = 0.0001
        max_rate = 0.02
        betas = min_rate + diffusion_times * (max_rate - min_rate)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1 - alpha_bars)
        return noise_rates, signal_rates
    return (linear_diffusion_schedule,)


@app.cell
def _(BATCH_SIZE, IMAGE_SIZE, linear_diffusion_schedule, torch, train_loader):
    images, _ = next(iter(train_loader))
    noises = torch.randn(size=(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))
    diffusion_times = torch.rand(size=(BATCH_SIZE, 1, 1, 1))
    noise_rates, signal_rates = linear_diffusion_schedule(diffusion_times)
    noisy_images = signal_rates * images + noise_rates * noises
    return (
        diffusion_times,
        images,
        noise_rates,
        noises,
        noisy_images,
        signal_rates,
    )


@app.cell
def _(noisy_images, plt):
    from torchvision.utils import make_grid
    grid = make_grid(noisy_images, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    return grid, make_grid


@app.cell
def _(nn, torch):
    import math

    class SinusoidalEmbedding(nn.Module):
        def __init__(self, num_frequencies=16):
            super().__init__()
            self.num_frequencies = num_frequencies
            frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
            self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

        def forward(self, x):
            """
            x: Tensor of shape (B, 1, 1, 1)
            returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
            """
            x = x.expand(-1, 1, 1, self.num_frequencies)
            sin_part = torch.sin(self.angular_speeds * x)
            cos_part = torch.cos(self.angular_speeds * x)
            return torch.cat([sin_part, cos_part], dim=-1)
    return SinusoidalEmbedding, math


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next we build the three new layers used in a UNet neural network, a popular architecture used in Diffusion Models.""")
    return


@app.cell
def _(F, nn, torch):
    class ResidualBlock(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, x):
            in_channels = x.shape[1]
            if in_channels != self.width:
                skip_conv = nn.Conv2d(in_channels, self.width, kernel_size=1).to(x.device)
            else:
                skip_conv = nn.Identity()

            norm = nn.BatchNorm2d(in_channels, affine=False).to(x.device)
            conv1 = nn.Conv2d(in_channels, self.width, kernel_size=3, padding=1).to(x.device)
            conv2 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=1).to(x.device)

            residual = skip_conv(x)
            x = norm(x)
            x = F.silu(conv1(x))  # Swish = SiLU
            x = conv2(x)
            return x + residual

    class DownBlock(nn.Module):
        def __init__(self, width, block_depth):
            super().__init__()
            self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(block_depth)])
            self.pool = nn.AvgPool2d(kernel_size=2)

        def forward(self, x, skips):
            for block in self.blocks:
                x = block(x)
                skips.append(x)
            x = self.pool(x)
            return x

    class UpBlock(nn.Module):
        def __init__(self, width, block_depth):
            super().__init__()
            self.block_depth = block_depth
            self.blocks = nn.ModuleList([ResidualBlock(width * 2 if i == 0 else width) for i in range(block_depth)])

        def forward(self, x, skips):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            for i in range(self.block_depth):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = self.blocks[i](x)
            return x
    return DownBlock, ResidualBlock, UpBlock


@app.cell
def _(DownBlock, F, ResidualBlock, SinusoidalEmbedding, UpBlock, nn, torch):
    class UNet(nn.Module):
        def __init__(self, image_size, num_channels, embedding_dim=32):
            super().__init__()
            self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
            self.num_channels = num_channels
            self.image_size = image_size
            self.embedding_dim = embedding_dim
            self.embedding = SinusoidalEmbedding(num_frequencies=16)
            self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

            self.down1 = DownBlock(32, block_depth=2)
            self.down2 = DownBlock(64, block_depth=2)
            self.down3 = DownBlock(96, block_depth=2)

            self.mid1 = ResidualBlock(128)
            self.mid2 = ResidualBlock(128)

            self.up1 = UpBlock(96, block_depth=2)
            self.up2 = UpBlock(64, block_depth=2)
            self.up3 = UpBlock(32, block_depth=2)

            self.final = nn.Conv2d(32, num_channels, kernel_size=1)
            nn.init.zeros_(self.final.weight)

        def forward(self, noisy_images, noise_variances):
            skips = []
            x = self.initial(noisy_images)

            noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
            noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.embedding_dim, self.embedding_dim), mode='nearest')
            x = torch.cat([x, self.embedding_proj(noise_emb)], dim=1)

            x = self.down1(x, skips)
            x = self.down2(x, skips)
            x = self.down3(x, skips)

            x = self.mid1(x)
            x = self.mid2(x)

            x = self.up1(x, skips)
            x = self.up2(x, skips)
            x = self.up3(x, skips)

            return self.final(x)
    return (UNet,)


@app.cell
def _(UNet, nn, torch):
    class DiffusionModel(nn.Module):
        def __init__(self, model, schedule_fn):
            super().__init__()
            self.network = model
            self.ema_network = UNet(model.image_size, model.num_channels, model.embedding_dim)
            self.ema_network.load_state_dict(model.state_dict())
            self.ema_decay = 0.999
            self.schedule_fn = schedule_fn
            self.normalizer_mean = 0.0
            self.normalizer_std = 1.0

        def set_normalizer(self, mean, std):
            self.normalizer_mean = mean
            self.normalizer_std = std

        def denormalize(self, x):
            return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

        def denoise(self, noisy_images, noise_rates, signal_rates, training):
            network = self.network if training else self.ema_network
            pred_noises = network(noisy_images, noise_rates ** 2)
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            return pred_noises, pred_images

        def reverse_diffusion(self, initial_noise, diffusion_steps):
            step_size = 1.0 / diffusion_steps
            current_images = initial_noise
            for step in range(diffusion_steps):
                t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
                noise_rates, signal_rates = self.schedule_fn(t)
                pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
                next_t = t - step_size
                next_noise_rates, next_signal_rates = self.schedule_fn(next_t)
                current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
            return pred_images

        def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
            if initial_noise is None:
                initial_noise = torch.randn((num_images, 1, image_size, image_size), device=next(self.parameters()).device)
            with torch.no_grad():
                return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

        def train_step(self, images, optimizer, loss_fn):
            images = (images - self.normalizer_mean) / self.normalizer_std
            noises = torch.randn_like(images)

            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = self.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            loss = loss_fn(pred_noises, noises)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                    ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)

            return loss.item()

        def test_step(self, images, loss_fn):
            images = (images - self.normalizer_mean) / self.normalizer_std
            noises = torch.randn_like(images)

            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = self.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            with torch.no_grad():
                pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
                loss = loss_fn(pred_noises, noises)

            return loss.item()
    return (DiffusionModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The training loop looks similar to the one used in the previous modules.""")
    return


@app.cell
def _():
    from tqdm import tqdm

    def train_diffusion(model, train_loader, val_loader, optimizer, loss_fn, epochs=10, device='cuda'):
        model.to(device)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, _  in loader_with_progress:
                images = images.to(device)
                loss = model.train_step(images, optimizer, loss_fn)
                train_losses.append(loss)

            avg_train_loss = sum(train_losses) / len(train_losses)

            model.eval()
            val_losses = []
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(device)
                loss = model.test_step(images, loss_fn)
                val_losses.append(loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            loader_with_progress.set_postfix(loss=f'{avg_train_loss:.4f}')
            #print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    return tqdm, train_diffusion


@app.cell
def _(
    DataLoader,
    DiffusionModel,
    IMAGE_SIZE,
    UNet,
    device,
    linear_diffusion_schedule,
    nn,
    test_loader,
    torch,
    train_dataset,
    train_diffusion,
    train_loader,
):
    NOISE_EMBEDDING_SIZE = 32
    NUM_CHANNELS = 1
    unet = UNet(IMAGE_SIZE, NUM_CHANNELS, NOISE_EMBEDDING_SIZE)
    diffusion_model = DiffusionModel(unet, linear_diffusion_schedule)

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    mean = 0.0
    std = 0.0
    count = 0
    train_loader_for_stats = DataLoader(train_dataset, batch_size=256)
    for imgs,_ in train_loader_for_stats:
        imgs = imgs.view(imgs.size(0), imgs.size(1), -1)
        mean += imgs.mean(dim=2).sum(0)
        std += imgs.std(dim=2).sum(0)
        count += imgs.size(0)
    mean /= count
    std /= count

    mean = mean.to(device)
    std = std.to(device)
    diffusion_model.set_normalizer(mean, std)
    train_diffusion(diffusion_model, train_loader, test_loader, optimizer, loss_fn, epochs=1, device=device)
    return (
        NOISE_EMBEDDING_SIZE,
        NUM_CHANNELS,
        count,
        diffusion_model,
        imgs,
        loss_fn,
        mean,
        optimizer,
        std,
        train_loader_for_stats,
        unet,
    )


@app.cell
def _(IMAGE_SIZE, diffusion_model, show_image):
    # Generate images
    diffusion_model.eval()
    samples = diffusion_model.generate(num_images=1, image_size=IMAGE_SIZE, diffusion_steps=1000)  # returns tensor in [0, 1]

    # Convert to numpy for plotting
    image = samples[0].cpu()

    # Plot
    show_image(image)
    return image, samples


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
