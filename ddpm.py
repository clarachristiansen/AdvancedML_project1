# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions.mixture_same_family import MixtureSameFamily
from tqdm import tqdm

# test = 0

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
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
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
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

        t = torch.randint(0, self.T, (batch_size,), device=x.device)
        epsilon = torch.randn_like(x)

        alpha_bar_t = self.alpha_cumprod[t].unsqueeze(1)

        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon

        t_normal = t.float().unsqueeze(1) / self.T
        epsilon_theta = self.network(x_t, t_normal)

        neg_elbo = F.mse_loss(epsilon_theta, epsilon, reduction='none')
        neg_elbo = neg_elbo.sum(dim=1)

        # global test
        # if test == 0:
        #     print(x.shape)
        #     print(t.shape)
        #     print(epsilon.shape)
        #     print(alpha_bar_t.shape)
        #     print(x_t.shape)
        #     print(t_normal.shape)
        #     print(epsilon_theta.shape)
        #     print(neg_elbo.shape)
        #     test += 1

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        device = self.alpha.device

        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape, device=device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T - 1, -1, -1):
            t_tensor = torch.full((shape[0],), t, device=device)
            t_normal = t_tensor.float().unsqueeze(1) / self.T

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]

            epsilon_theta = self.network(x_t, t_normal)

            mean = (1/torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t

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

def train(model, optimizer, data_loader, epochs, device):
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

class Unet(torch.nn.Module):
    """
    A simple U-Net architecture for MNIST that takes an input image and time
    """
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid

    # # Parse arguments
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'sampleMnist'], help='what to do when running the script (default: %(default)s)')
    # parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    # parser.add_argument('--model', type=str, default='Week3/model.pt', help='file to save model to or load model from (default: %(default)s)')
    # parser.add_argument('--samples', type=str, default='Week3/samples.png', help='file to save samples in (default: %(default)s)')
    # parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    # parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    # parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: %(default)s)')
    # parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = ['compareMnist', 'mnist', 'UnetMnistmodel.pt', 'Unetsamples.png', 'cpu', 64, 100, 1e-3] #parser.parse_args()
    print('# Options')
    for value in args:
        print(value)

    transform = transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Lambda(lambda x : x + torch.rand(x.shape)/255.0), 
                                                transforms.Lambda(lambda x : (x-0.5)*2.0),
                                                transforms.Lambda(lambda x : x.flatten())])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = True, download = True, transform = transform), batch_size=args[5], shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = False, download = True, transform = transform), batch_size=args[5], shuffle=True)

    # Get the dimension of the dataset
    D = next(iter(train_loader))[0].shape[1]

    # Define the network
    num_hidden = 64
    network = Unet() #FcNetwork(D, num_hidden)

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args[4])

    # Choose mode to run
    if args[0] == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args[7])

        # Train model
        train(model, optimizer, train_loader, args[6], args[4])

        # Save model
        torch.save(model.state_dict(), args[2])

    elif args[0] == 'sampleMnist':
        model.load_state_dict(torch.load(args[2], map_location=torch.device(args[4]),weights_only=True))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((64,D))).cpu() 
        
        samples = samples.view(-1, 1, 28, 28)

        # Transform the samples back to the original space
        samples = samples/2 + 0.5

        grid = make_grid(samples, nrow=8, padding=2)
        save_image(grid, args[1] + args[3])

    elif args[0] == 'compareMnist':
        model.load_state_dict(torch.load('UnetMnistmodel.pt', map_location=torch.device(args[4]),weights_only=True))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples1 = (model.sample((4,D))).cpu()
        samples1 = samples1.view(-1, 1, 28, 28)
        samples1 = samples1/2 + 0.5
        print("hej1")
        
        #model = latentDDPM(network, T=T).to(args[4])
        model.load_state_dict(torch.load('UnetMnistmodel.pt', map_location=torch.device(args[4]),weights_only=True))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples2 = (model.sample((4,D))).cpu()
        samples2 = samples2.view(-1, 1, 28, 28)
        samples2 = samples2/2 + 0.5
        print("hej2")

        #model = VAE(network, T=T).to(args[4])
        model.load_state_dict(torch.load('UnetMnistmodel.pt', map_location=torch.device(args[4]),weights_only=True))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples3 = (model.sample((4,D))).cpu()
        samples3 = samples3.view(-1, 1, 28, 28)
        samples3 = samples3/2 + 0.5
        print("hej3")

        combine = torch.cat([samples1, samples2, samples3], dim=0)
        grid = make_grid(combine, nrow=4, padding=2)
        save_image(grid, "MnistComparison.png")