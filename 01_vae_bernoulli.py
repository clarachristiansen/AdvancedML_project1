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

### EXERCISE 1.6
class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Mixture of Gaussians prior p(z) = sum_k pi_k N(z | mu_k, diag(sigma_k^2))

        M: latent dimension
        K: number of mixture components
        """
        super().__init__()
        self.M = M
        self.K = K

        # mixture weights (logits); start uniform
        self.logits = nn.Parameter(torch.zeros(K))

        # means
        self.loc = nn.Parameter(torch.randn(K, M) * 0.01)

        # scales; parameterize via log_scale for positivity
        self.log_scale = nn.Parameter(torch.zeros(K, M))

    def forward(self):
        mix = td.Categorical(logits=self.logits)  # shape: [K]
        comp = td.Independent(
            td.Normal(loc=self.loc, scale=torch.exp(self.log_scale)),
            1
        )  # batch shape [K], event shape [M]
        return td.MixtureSameFamily(mix, comp)


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
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)




class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

### EXERCISE 1.7
class GaussianDecoderFixedVar(nn.Module):
    def __init__(self, decoder_net, sigma=0.1, learn_sigma=False):
        super(GaussianDecoderFixedVar, self).__init__()
        self.decoder_net = decoder_net

        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(float(sigma)).log())
        else:
            self.register_buffer("log_sigma", torch.tensor(float(sigma)).log())

    def forward(self, z):
        mu = self.decoder_net(z)  # [B,28,28]
        sigma = torch.exp(self.log_sigma)  # scalar
        return td.Independent(td.Normal(loc=mu, scale=sigma), 2)

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
    def __init__(self, prior, decoder, encoder):
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
        try:
            p = self.prior()
        except:
            p = self.prior

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
        elbo = (recon - kl).mean(dim=0).mean()
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)
    

def train(model, optimizer, data_loader, epochs, device):
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

def data_to_plot_posterior(model, data_loader, samples_per_x, pca, device, max_points=int(1e5)):
    """
    """
    model.eval()
    zs = []
    ys = []
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample(sample_shape=torch.Size([samples_per_x]))
            S, B, M = z.shape
            z = z.permute(1, 0, 2).reshape(B * S, M)
            zs.append(z.cpu())

            ys.append(y.repeat_interleave(samples_per_x))
            total += B * S
            if (max_points is not None) and (total >= max_points):
                    break
    zs = torch.cat(zs, dim=0)[:max_points] if max_points is not None else torch.cat(zs, dim=0)
    zs = zs.numpy()
    ys = torch.cat(ys, dim=0)[:zs.shape[0]].numpy()
    if zs.shape[1] > 2:
        zs = pca.fit_transform(zs)
    return zs, ys

def plot_posterior(z, title, name):
    plt.figure(figsize=(6,6))
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10', s=5, alpha=0.8)
    plt.colorbar()
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title(title)
    plt.savefig(f'{name}.png', dpi=1000)
    plt.show()

def plot_all(z_prior, z_posterior, y_posterior, name):
    fig, ax = plt.subplots(2, 1, figsize=(8, 15), sharex=True, sharey=True, constrained_layout=True)
    ax[0].scatter(z_prior[:, 0], z_prior[:, 1], s=5, alpha=0.8)
    ax[0].set(xlabel='z1', ylabel='z2')
    ax[0].set_title(f'Prior: {name}')

    sc = ax[1].scatter(z_posterior[:, 0], z_posterior[:, 1], c=y_posterior, cmap='tab10', s=5, alpha=0.8)
    #cbar = fig.colorbar(sc, ax=ax, ticks=range(10))
    #cbar.set_label('Class')
    ax[1].set(xlabel='z1', ylabel='z2')
    ax[1].set_title(f'Aggregated posterior')
    cbar = fig.colorbar(
        sc, ax=ax, orientation='horizontal',
        ticks=range(10),
        fraction=0.06, pad=0.01, aspect=40
    )
    cbar.set_ticks(torch.arange(10) * 0.9 + 0.5)
    cbar.set_ticklabels(range(10))
    cbar.set_label("Class")
    #fig.tight_layout()
    fig.savefig(f'{name}.png', dpi=1000)

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'sample_mean', 'evaluate', 'plot_posterior', 'plot_all'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='what prior to use (default: %(default)s)')
    parser.add_argument('--mask', type=str, default='normal', choices=['normal', 'random'], help='what mask to use (default: %(default)s)')
    parser.add_argument('--data', type=str, default='binary', choices=['binary', 'continuous_fixed', 'continuous_pixel'], help='whether to use binary or continuous MNIST (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
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


    # Load MNIST as binarized at 'threshold' and create data loaders
    threshold = 0.5
    if args.data == 'continuous_fixed' or args.data == 'continuous_pixel':
        threshold = 0.0
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim

    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MoGPrior(M, K=10)
    elif args.prior == 'flow':
        # Define prior distribution
        transformations = []
        num_transformations = 16
        num_hidden = 256
        M = args.latent_dim
        latent_dim = M
        base = GaussianBase(latent_dim)

        if args.mask == 'normal':
            # Make a mask that is 1 for the first half of the features and 0 for the second half
            mask = torch.zeros((latent_dim,))
            mask[latent_dim//2:] = 1

        for i in range(num_transformations):
            if args.mask == 'random':
                mask = torch.randint(0, 2, (latent_dim,), dtype=torch.float32)
            elif args.mask == 'normal':
                mask = (1-mask) # Flip the mask
            
            scale_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
            translation_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
            
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        prior = Flow(base, transformations)


    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
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
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    if args.data == 'binary':
            decoder = BernoulliDecoder(decoder_net)
    elif args.data == 'continuous_fixed':
        decoder = GaussianDecoderFixedVar(decoder_net, sigma=0.1, learn_sigma=True)
    elif args.data == 'continuous_pixel':
        decoder = GaussianDecoderPixelVar(decoder_net, init_sigma=0.1)
        
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples[:-4] + f'_{args.prior}_{args.data}.png')
    elif args.mode == "sample_mean":
        model.load_state_dict(torch.load(args.model, map_location=device))

        # Generate samples
        model.eval()
        with torch.no_grad():
            if args.prior == 'flow':
                z = model.prior.sample((64,))
            else:
                z = model.prior().sample((64,))  # [64, M]
            dist = model.decoder(z) # [64, 28, 28]
            x_sample = dist.sample() # [64, 28, 28]
            x_mean = dist.mean  # [64, 28, 28]

            x_sample = x_sample.clamp(0, 1).unsqueeze(1)  # [64,1,28,28]
            x_mean   = x_mean.clamp(0, 1).unsqueeze(1)

            save_image(x_sample, args.samples[:-4] + f'_{args.prior}_{args.data}_sample.png')
            save_image(x_mean, args.samples[:-4] + f'_{args.prior}_{args.data}_mean.png')

    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(args.model, map_location=device))
        avg_elbo = evaluate_elbo(model, mnist_test_loader, device)
        #print("Test ELBO (nats / example):", avg_elbo)
        print(f"RESULT seed={args.seed} test_elbo={avg_elbo}")
    elif args.mode == 'plot_posterior':
        model.load_state_dict(torch.load(args.model, map_location=device))
        z_posterior, y_posterior = data_to_plot_posterior(model, mnist_test_loader, 10, device=device)
        plot_posterior(z_posterior, title='Aggregate posterior samples colored by true label', name='plot_posterior')
    elif args.mode == 'plot_all':
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        n_samples = 100000
        with torch.no_grad():
            if args.prior == 'flow':
                prior_points = model.prior.sample(torch.Size([n_samples]))
            else:
                prior_points = model.prior().sample(torch.Size([n_samples]))
            if prior_points.shape[1] > 2:
                pca = PCA(n_components=2)
            z_prior = pca.fit_transform(prior_points.cpu().detach().numpy())

            z_posterior, y_posterior = data_to_plot_posterior(model, mnist_test_loader, 10, pca, device=device)
            
        plot_all(z_prior, z_posterior, y_posterior, name='Flow')