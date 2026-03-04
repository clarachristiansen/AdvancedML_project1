# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import betaVAE 

class DDPM(nn.Module):
    def __init__(self, network, vae, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.vae = vae
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    def encode(self, x):
        with torch.no_grad():
            z = self.vae.encoder(x).rsample()
        return z
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        batch_size = x.shape[0]

        #Encode for training
        z0 = self.encode(x)

        t = torch.randint(0, self.T, (batch_size,), device=x.device)
        epsilon = torch.randn_like(z0)

        alpha_bar_t = self.alpha_cumprod[t].unsqueeze(1)

        z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * epsilon

        t_normal = t.float().unsqueeze(1) / self.T
        epsilon_theta = self.network(z_t, t_normal)

        loss = F.mse_loss(epsilon_theta, epsilon, reduction='none')
        return loss.sum(dim=1)

    def sample(self, num):
        """
        Sample from the model.

        Parameters:
        num: [int]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        device = self.alpha.device
        z_dim = self.vae.encoder.encoder_net[-1].out_features // 2

        # Sample z_t for t=T (i.e., Gaussian noise)
        z_t = torch.randn(num, z_dim, device=device)

        # Sample z_t given z_{t+1} until z_0 is sampled
        for t in range(self.T - 1, -1, -1):
            t_tensor = torch.full((num,), t, device=device)
            t_normal = t_tensor.float().unsqueeze(1) / self.T

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]

            epsilon_theta = self.network(z_t, t_normal)

            mean = (1/torch.sqrt(alpha_t)) * (z_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)

            if t > 0:
                noise = torch.randn_like(z_t)
                z_t = mean + torch.sqrt(beta_t) * noise
            else:
                z_t = mean
        
        #Decode for sampling
        with torch.no_grad():
            x = self.vae.decoder(z_t)
            x_samples = x.mean

        return x_samples

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()

def train_ddpm(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(),
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(),
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid

    #          mode     data          model       sample save    device batch epo  lr   beta    prior
    args = ['sample', 'mnist', 'Latentmodel', 'Latentsamples.png', 'cuda', 32, 150, 3e-4, 1.0, "gaus"]

    transform = transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Lambda(lambda x : x + torch.rand(x.shape)/255.0), 
                                                transforms.Lambda(lambda x : (x-0.5)*2.0),
                                                transforms.Lambda(lambda x : x.flatten())])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = True, download = True, transform = transform), batch_size=args[5], shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = False, download = True, transform = transform), batch_size=args[5], shuffle=True)

    # Get the dimension of the dataset
    D = next(iter(train_loader))[0].shape[1]

    # Define the network
    latent_dim = 64
    hidden_dim = 512 #This num hidden and the one in the en- and decoders does not need to match but that is how i have implemented it right now
    network = FcNetwork(latent_dim, hidden_dim) 

    # Set the number of steps in the diffusion process
    T = 500

    if args[9] == "flow":
        # Define prior distribution
        transformations = []
        num_transformations = 16
        num_hidden = 512
        base = betaVAE.GaussianBase(latent_dim)

        mask = torch.zeros((latent_dim,))
        mask[latent_dim//2:] = 1

        for i in range(num_transformations):
            mask = (1-mask) # Flip the mask
            
            scale_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
            translation_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
            
            transformations.append(betaVAE.MaskedCouplingLayer(scale_net, translation_net, mask))
        prior = betaVAE.Flow(base, transformations)
    elif args[9] == "gaus":
        prior = betaVAE.GaussianPrior(latent_dim)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, latent_dim*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 512),
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
    decoder = betaVAE.GaussianDecoderPixelVar(decoder_net, init_sigma=0.1)
    encoder = betaVAE.GaussianEncoder(encoder_net)
    vae = betaVAE.VAE(prior, decoder, encoder, beta=args[8])
    vae.load_state_dict(torch.load(f"models/{args[9]}VAE{args[8]}.pt", map_location=torch.device(args[4]),weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Define model
    model = DDPM(network, vae, T=T).to(args[4])

    # Choose mode to run
    if args[0] == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args[7])

        # Train model
        train_ddpm(model, optimizer, train_loader, args[6], args[4])

        # Save model
        torch.save(model.state_dict(), f"models/{args[9]}{args[2]}{args[8]}.pt")
    if args[0] == 'sample':
        model.load_state_dict(torch.load(f"models/{args[9]}{args[2]}{args[8]}.pt", map_location=torch.device(args[4]),weights_only=True))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64)
        samples = samples.view(-1, 1, 28, 28)
        print("min/max pre")
        print(samples.min(), samples.max())
        samples = samples / 2 + 0.5
        print("min/max post")
        print(samples.min(), samples.max())

        grid = make_grid(samples, nrow=8)
        save_image(grid, f"Samples/{args[9]}{args[1]}{args[8]}{args[3]}")