import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 8: Practical 1 - Energy Based Methods""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        First, some math... The derivations below are somewhat complex, so do not worry if you don't follow everything. Luckily, the final loss function is simple and intuitive. The derivation is not necessary to understand the code, but it is useful to know where the loss function comes from.

        This follows a nice derivation of Contrastive Divergence found in https://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf. If you reference the above, keep in mind a change of notation: our $E$ is the same as $-log(f)$ in the derivation (equivalently $f = e^{-E}$).
        """
    )
    return


app._unparsable_cell(
    r"""
    We will first derive a loss function for our energy based models using maximum likelyhood. Later we will see that the resulting loss function makes sense even if we don't transform the energy into a probability distribution.
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Start with an energy defined for all inputs $x$ and dependent on parameters $w$: $E(x, w)$.

        We can transform it into a probability by normalizing it as follows:

        $p(x,w) = \dfrac{1}{Z(w)} e^{-E(x,w)}$,

        where

        $Z(w) = \int e^{-E(x,w)}$.

        Here we are assuming that $x$ is a continuous variable, so the integration is analogous to summation in the case of $x$ being discrete valued. If you are not familiar with integration, just think about it being aproximated by "summing" the energy values over the possible values of $x$. This should actually look familiar.  Remember the softmax activation?  In the discrete case, this is how we usually turn multiple outputs into a probability distribution used for classification in a neural network.

        The difference is that we are only using this form now to derive an appropriate loss function - with  energy models, our loss function will work directly with unnormalized energy values.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that we have a probability function, we can use the usual Maximum Likelyhood Estimation.

        $loss = -\mathbb{E}_{x \sim data} \log p(x,w) = -\dfrac{1}{N} \sum \log p(x_i, w) = -\dfrac{1}{N} \sum \log \left( \dfrac{1}{Z(w)} e^{-E(x_i,w)} \right) = \dfrac{1}{N} \sum \left( \log Z(w) -E(x_i, w) \right)$.

        Since $Z$ doesn't depend on $i$ or $x$ this simplifies to

        $loss = \log Z(w) + \dfrac{1}{N} \sum E(x_i, w)$.

        To train a neural network using gradient descent requires calculating the gradient of the loss with respect to the parameters $w$:

        $\nabla_w loss = \nabla_w \log Z(w) + \dfrac{1}{N} \sum \nabla_w E(x_i, w) =  \nabla_w \log Z(w)  + \mathbb{E}_{x \sim data} \nabla_w E(x, w)$.

        We will obtain a form for the first term in terms of an expectation as well.

        $\nabla_w \log Z(w) = \dfrac{1}{Z(w)} \nabla_w Z(w) = \dfrac{1}{Z(w)} \nabla_w \int e^{-E(x,w)} =  -\dfrac{1}{Z(w)} \int e^{- E(x,w)} \nabla_w E(x,w) = -\dfrac{1}{Z(w)} \int e^{- E(x,w)} \nabla_w E(x,w) = -\int p(x,w) \nabla_w E(x,w) = - \mathbb{E}_{x \sim model} \nabla_w E(x,w)$

        and the full gradient becomes

        $loss = \mathbb{E}_{x \sim data} \nabla_w E(x, w) - \mathbb{E}_{x \sim model} \nabla_w E(x,w)$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In other words we are trying to find parameters $w$ that decrease the energy on the data and increase the energy on the model.  This is a form of contrastive divergence, where we are trying to minimize the difference between the data and the model distributions. It penalizes $x$ values "away" from the data distribution.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now let's implement an Energy Based model using the loss function we derived. First the usual data preparation steps...""")
    return


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


@app.cell
def _(torch):
    # Check which GPU is available
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    return (device,)


@app.cell
def _(torchvision, transforms):
    # Prepare the Data
    transform = transforms.Compose([transforms.Pad(2, -1), transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return test_dataset, train_dataset, transform


@app.cell
def _(DataLoader, test_dataset, train_dataset):
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return BATCH_SIZE, test_loader, train_loader


@app.cell
def _(torch):
    def generate_samples(model, inp_imgs, steps, step_size, noise_std):
        model.eval()
        imgs_per_step = []

        inp_imgs = inp_imgs.clone().detach().requires_grad_(True)

        for _ in range(steps):
            # Add noise
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs.data += noise
            inp_imgs.data.clamp_(-1.0, 1.0)

            # Compute gradients
            model.zero_grad()
            out_score = -model(inp_imgs).sum()  # negative energy
            out_score.backward()

            # Get gradients and apply Langevin update
            grads = inp_imgs.grad.data.clamp_(-0.03, 0.03)
            inp_imgs.data += -step_size * grads
            inp_imgs.data.clamp_(-1.0, 1.0)

            # Optionally store intermediate results
            imgs_per_step.append(inp_imgs.detach().clone())

            # Clear gradients for next step
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()

        return inp_imgs.detach(), imgs_per_step
    return (generate_samples,)


@app.cell
def _(generate_samples, np, torch):
    import random

    class Buffer:
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.examples = [
                torch.rand((1, 1, 32, 32)) * 2 - 1  # Shape: (1, 1, 32, 32), range: [-1, 1]
                for _ in range(128)
            ]

        def sample_new_exmps(self, steps, step_size, noise):
            n_new = np.random.binomial(128, 0.05)

            # Generate new random images
            if n_new > 0:
                rand_imgs = torch.rand((n_new, 1, 32, 32)) * 2 - 1
            else:
                rand_imgs = torch.empty((0, 1, 32, 32))

            # Sample old images
            old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)

            # Combine new and old
            if n_new > 0:
                inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0)
            else:
                inp_imgs = old_imgs

            # Run Langevin dynamics
            new_imgs, _ = generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size, noise_std=noise)

            # Update buffer
            self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
            self.examples = self.examples[:8192]

            return new_imgs
    return Buffer, random


@app.cell
def _(torch):
    # Swish activation function
    def swish(x):
        return x * torch.sigmoid(x)
    return (swish,)


@app.cell
def _(nn, swish):
    class EBMModel(nn.Module):
        def __init__(self):
            super(EBMModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 2 * 2, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = swish(self.conv1(x))
            x = swish(self.conv2(x))
            x = swish(self.conv3(x))
            x = swish(self.conv4(x))
            x = self.flatten(x)
            x = swish(self.fc1(x))
            return self.fc2(x)

    # Example instantiation
    # model = EBM()
    # print(model)
    return (EBMModel,)


@app.cell
def _(Buffer, nn, torch):
    from collections import defaultdict

    class Metric:
        def __init__(self):
            self.reset()

        def update(self, val):
            self.total += val.item()
            self.count += 1

        def result(self):
            return self.total / self.count if self.count > 0 else 0.0

        def reset(self):
            self.total = 0.0
            self.count = 0

    class EBM(nn.Module):
        def __init__(self, model, alpha, steps, step_size, noise):
            super().__init__()
            self.model = model
            self.buffer = Buffer(self.model)
            self.alpha = alpha
            self.steps = steps
            self.step_size = step_size
            self.noise = noise

            self.loss_metric = Metric()
            self.reg_loss_metric = Metric()
            self.cdiv_loss_metric = Metric()
            self.real_out_metric = Metric()
            self.fake_out_metric = Metric()

        def metrics(self):
            return {
                "loss": self.loss_metric.result(),
                "reg": self.reg_loss_metric.result(),
                "cdiv": self.cdiv_loss_metric.result(),
                "real": self.real_out_metric.result(),
                "fake": self.fake_out_metric.result()
            }

        def reset_metrics(self):
            for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                      self.real_out_metric, self.fake_out_metric]:
                m.reset()

        def train_step(self, real_imgs, optimizer):
            real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
            real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

            fake_imgs = self.buffer.sample_new_exmps(
                steps=self.steps, step_size=self.step_size, noise=self.noise
            )

            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
            inp_imgs.requires_grad = True

            optimizer.zero_grad()
            out_scores = self.model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

            cdiv_loss = fake_out.mean() - real_out.mean()
            reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss

            loss.backward()
            optimizer.step()

            self.loss_metric.update(loss)
            self.reg_loss_metric.update(reg_loss)
            self.cdiv_loss_metric.update(cdiv_loss)
            self.real_out_metric.update(real_out.mean())
            self.fake_out_metric.update(fake_out.mean())

            return self.metrics()

        def test_step(self, real_imgs):
            batch_size = real_imgs.shape[0]
            fake_imgs = torch.rand((batch_size, 1, 32, 32)) * 2 - 1
            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

            with torch.no_grad():
                out_scores = self.model(inp_imgs)
                real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
                cdiv = fake_out.mean() - real_out.mean()

            self.cdiv_loss_metric.update(cdiv)
            self.real_out_metric.update(real_out.mean())
            self.fake_out_metric.update(fake_out.mean())

            return {
                "cdiv": self.cdiv_loss_metric.result(),
                "real": self.real_out_metric.result(),
                "fake": self.fake_out_metric.result()
            }
    return EBM, Metric, defaultdict


@app.cell
def _(EBM, EBMModel, test_loader, torch, train_loader):
    # Initialize model, optimizer
    base_model = EBMModel()
    ebm = EBM(base_model, alpha=0.1, steps=60, step_size=10.0, noise=0.005)
    optimizer = torch.optim.Adam(ebm.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(1):
        ebm.reset_metrics()
        for batch in train_loader:
            real_imgs = batch[0]
            metrics = ebm.train_step(real_imgs, optimizer)

        print(f"Epoch {epoch+1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        # Validation step
        ebm.reset_metrics()
        for batch in test_loader:
            real_imgs = batch[0]
            val_metrics = ebm.test_step(real_imgs)

        print(f"Validation - " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
    return (
        base_model,
        batch,
        ebm,
        epoch,
        metrics,
        optimizer,
        real_imgs,
        val_metrics,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
