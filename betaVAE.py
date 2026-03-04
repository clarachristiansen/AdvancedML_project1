# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """

        scale = self.scale_net(z * self.mask)
        scale = torch.tanh(scale)  # stability
        translation = self.translation_net(z * self.mask)

        x = (z * self.mask) + (1 - self.mask) * (z * torch.exp(scale) + translation)

        log_det_J = ((1-self.mask) * scale).flatten(start_dim=1).sum(dim=1)

        return x, log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        scale = self.scale_net(x * self.mask)
        scale = torch.tanh(scale)  # stability
        translation = self.translation_net(x * self.mask)
        z = (x * self.mask) + (1 - self.mask) * ((x - translation) * torch.exp(-scale))
        log_det_J = -((1 - self.mask) * scale).flatten(start_dim=1).sum(dim=1)
        return z, log_det_J

class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0] # OBS why? Maybe Gaussian error
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)

        std = torch.clamp(std, min=-10, max=10) #It seems like we have numerical instability

        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class GaussianDecoderPixelVar(nn.Module):
    def __init__(self, decoder_net, init_sigma=0.1):
        super().__init__()
        self.decoder_net = decoder_net
        self.log_sigma = nn.Parameter(torch.log(torch.ones(28, 28) * init_sigma))

    def forward(self, z):
        mu = self.decoder_net(z)  # [B,28,28]
        sigma = torch.exp(self.log_sigma).unsqueeze(0)  # [1,28,28] -> broadcast to [B,28,28]
        return td.Independent(td.Normal(loc=mu, scale=sigma), 2)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, beta=1.0):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x, n_samples=3):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample((n_samples,))  # [S, B, M]
        # Before exercise 1.6: elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)

        # Expand x to match sample dimension
        #xS = x.unsqueeze(0).expand(n_samples, *x.shape)  # [S, B, ...]

        # Reconstruction term
        recon = self.decoder(z).log_prob(x)  # [S, B]

        # Prior ### OBS:
        p = self.prior if hasattr(self.prior, "log_prob") else self.prior()

        # KL term
        try:
            # closed-form KL only depends on q and p, not z
            kl = td.kl_divergence(q, p)       # [B]
            kl = kl.unsqueeze(0).expand(n_samples, -1)  # [S, B]
        except Exception:
            log_qz = q.log_prob(z)
            z_flat = z.reshape(-1, z.shape[-1])
            log_pz = p.log_prob(z_flat).view(z.shape[0], z.shape[1])

            kl = log_qz - log_pz  # [S, B]

            #kl = q.log_prob(z) - p.log_prob(z)  # [S, B]

        # Average over samples, then batch
        elbo = (recon - self.beta * kl).mean(dim=0).mean()
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        if hasattr(self.prior, "sample"):  # Flow prior
            z = self.prior.sample((n_samples,))
        else:  # Base distribution prior
            z = self.prior().sample((n_samples,))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)
    
def train_vae(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate_elbo(model, data_loader, device):
    """
    Evaluate the ELBO of a VAE model on a given dataset.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.

    Returns:
    avg_elbo: [float]
        The average ELBO (in nats per example) over the dataset.
    """
    model.eval()
    total_elbo = 0.0
    total_examples = 0

    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            batch_size = x.size(0)
            total_elbo += model.elbo(x).item() * batch_size
            total_examples += batch_size

    avg_elbo = total_elbo / total_examples
    return avg_elbo

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'sample_mean', 'evaluate', 'plot_posterior', 'plot_all'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaus', choices=['gaus', 'flow'], help='What prior to use (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1.0, metavar='B', help='value for beta in the beta-VAE (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = torch.device(args.device) 
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)


    transform = transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Lambda(lambda x : x + torch.rand(x.shape)/255.0), 
                                                transforms.Lambda(lambda x : (x-0.5)*2.0),
                                                transforms.Lambda(lambda x : x.squeeze())])
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = True, download = True, transform = transform), batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = False, download = True, transform = transform), batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim

    if args.prior == "flow":
        # Define prior distribution
        transformations = []
        num_transformations = 16
        num_hidden = 512
        M = args.latent_dim
        latent_dim = M
        base = GaussianBase(latent_dim)

        mask = torch.zeros((latent_dim,))
        mask[latent_dim//2:] = 1

        for i in range(num_transformations):
            mask = (1-mask) # Flip the mask
            
            scale_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
            translation_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
            
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        prior = Flow(base, transformations)
    elif args.prior == "gaus":
        prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 28*28),
        nn.Tanh(),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = GaussianDecoderPixelVar(decoder_net, init_sigma=0.1)
        
    encoder = GaussianEncoder(encoder_net)

    beta = args.beta
    model = VAE(prior, decoder, encoder, beta).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train model
        train_vae(model, optimizer, mnist_train_loader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), f"models/{args.prior}VAE{beta}.pt")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f"models/{args.prior}VAE{beta}.pt", map_location=device,weights_only=True))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            samples = samples / 2 + 0.5
            save_image(samples.view(64, 1, 28, 28), f"Samples/{args.prior}{args.beta}{args.samples}")

    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(f"models/{args.prior}VAE{beta}.pt", map_location=device))
        avg_elbo = evaluate_elbo(model, mnist_test_loader, device)
        #print("Test ELBO (nats / example):", avg_elbo)
        print(f"RESULT seed={args.seed} test_elbo={avg_elbo}")

        z = []
        for x,_ in mnist_train_loader:
            z.append(model.encoder(x.to(device)).rsample().detach().cpu())
        z = torch.cat(z)

        print(z.mean())
        print(z.std())

    # python betaVAE.py train --prior flow --samples Samples.png --device cpu --batch-size 128 --epochs 80 --latent-dim 64 --beta 1.0
    # python betaVAE.py sample --prior flow --samples Samples.png --device cpu --batch-size 128 --epochs 80 --latent-dim 64 --beta 1.0
    # python betaVAE.py evaluate --prior flow --samples Samples.png --device cpu --batch-size 128 --epochs 80 --latent-dim 64 --beta 1.0