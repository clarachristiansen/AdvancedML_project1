import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import ddpm
import betaVAE
import VAEddpm

import fid

def loadDDPM(device):
    network = ddpm.Unet()
    T = 1000
    model = ddpm.DDPM(network, T=T).to(device)
    model.load_state_dict(torch.load('Project1/Unetmodel.pt', map_location=torch.device(device),weights_only=True))
    return model

def loadLatentDDPM(beta, device):
    latent_dim = 64
    hidden_dim = 512 
    network = VAEddpm.FcNetwork(latent_dim, hidden_dim) 

    # Set the number of steps in the diffusion process
    T = 500
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
    vae = betaVAE.VAE(prior, decoder, encoder, beta=beta)
    vae.load_state_dict(torch.load(f"Project1/gausVAE{beta}.pt", map_location=torch.device(device),weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Define model
    model = VAEddpm.DDPM(network, vae, T=T).to(device)
    model.load_state_dict(torch.load(f"Project1/gausLatentmodel{beta}.pt", map_location=torch.device(device),weights_only=True))
    return model, vae

def loadVAE(device):
    # Define prior distribution
    transformations = []
    num_transformations = 16
    num_hidden = 512
    latent_dim = 64
    base = betaVAE.GaussianBase(latent_dim)

    mask = torch.zeros((latent_dim,))
    mask[latent_dim//2:] = 1

    for i in range(num_transformations):
        mask = (1-mask) # Flip the mask
        
        scale_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
        translation_net = nn.Sequential(nn.Linear(latent_dim, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, latent_dim))
        
        transformations.append(betaVAE.MaskedCouplingLayer(scale_net, translation_net, mask))
    prior = betaVAE.Flow(base, transformations)

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
    model = betaVAE.VAE(prior, decoder, encoder, 1.0).to(device)
    model.load_state_dict(torch.load(f"Project1/flowVAE1.0.pt", map_location=device,weights_only=True))
    return model

def evalFid(model, model_name, real_batch, num_samp, D, device):
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        if model_name == "DDPM":
            batch = (model.sample((num_samp,D))).to(device)
        else:
            batch = model.sample(num_samp).to(device)
    t2 = time.time()
    batch = batch.view(-1, 1, 28, 28)
    batch = batch / 2 + 0.5
    print(batch.shape)

    print(f"Time for {num_samp} samples for model {model_name} = {t2-t1}")
    print(f"{(t2-t1)/num_samp} seconds per sample")

    fid_res = fid.compute_fid(real_batch, batch, device)
    print(f"Fid for {model_name}: {fid_res}")

#        mode       device  batch 
args = ['plot', 'cuda', 10000]

#load data
transform = transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Lambda(lambda x : x + torch.rand(x.shape)/255.0), 
                                            transforms.Lambda(lambda x : (x-0.5)*2.0),
                                            transforms.Lambda(lambda x : x.flatten())])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = True, download = True, transform = transform), batch_size=args[2], shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train = False, download = True, transform = transform), batch_size=args[2], shuffle=True)
D = next(iter(train_loader))[0].shape[1]

# Choose mode to run
if args[0] == "computeFid":
    import time

    real_batch = next(iter(train_loader))[0]
    real_batch = real_batch.view(-1, 1, 28, 28).to(args[1])
    real_batch = real_batch / 2 + 0.5
    print(real_batch.shape)

    # DDPM model
    model = loadDDPM(args[1])
    evalFid(model, "DDPM", real_batch, args[2], D, args[1])

    # latent ddpm models
    for beta in [1e-6,0.5,1.0,2.0]:
        model, _ = loadLatentDDPM(beta, args[1])
        evalFid(model, f"LatentDDPM{beta}", real_batch, args[2], D, args[1])
    
    # VAE model
    model = loadVAE(args[1])
    evalFid(model, "VAE", real_batch, args[2], D, args[1])

elif args[0] == 'mcompare':
    model1 = loadDDPM(args[1])
    model2, _ = loadLatentDDPM(1.0, args[1])
    model3 = loadVAE(args[1])
    for i in range(10):
        # DDPM model
        with torch.no_grad():
            batch1 = (model1.sample((4,D))).to(args[1])
        batch1 = batch1.view(-1, 1, 28, 28)
        batch1 = batch1 / 2 + 0.5

        # latent ddpm models
        with torch.no_grad():
            batch2 = (model2.sample(4)).to(args[1])
        batch2 = batch2.view(-1, 1, 28, 28)
        batch2 = batch2 / 2 + 0.5
        
        # VAE model
        with torch.no_grad():
            batch3 = (model3.sample(4)).to(args[1])
        batch3 = batch3.view(-1, 1, 28, 28)
        batch3 = batch3 / 2 + 0.5

        combine = torch.cat([batch1, batch2, batch3], dim=0)
        grid = make_grid(combine, nrow=4, padding=2)
        save_image(grid, f"Project1/mcompares/MnistComparison{i}.png")

elif args[0] == "plot":
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from torchvision import datasets, transforms
    model, vae = loadLatentDDPM(1.0, args[1])

    num_samples = 5000

    #Posterior
    z_post = []
    with torch.no_grad():
        for x in iter(train_loader):
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(args[1])
            q = vae.encoder(x)
            z = q.rsample()
            z_post.append(z)

            if len(torch.cat(z_post)) >= num_samples:
                break

    z_post = torch.cat(z_post)[:num_samples].cpu()
    print(z_post.shape)

    #prior
    with torch.no_grad():
        z_prior = vae.prior().sample(torch.Size([num_samples])).cpu()
    print(z_prior.shape)

    #latent DDPM samples
    with torch.no_grad():
        z_ddpm = model.sample(num_samples, decode=False).cpu()
    print(z_ddpm.shape)

    #pca, ugly implementation but i could not find a better one
    all_z = torch.cat([z_post, z_prior, z_ddpm], dim=0).numpy()
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_z)
    
    z_post_2d  = all_2d[:num_samples]
    z_prior_2d = all_2d[num_samples:2*num_samples]
    z_ddpm_2d  = all_2d[2*num_samples:]

    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    plt.scatter(z_post_2d[:,0], z_post_2d[:,1], s=5, alpha=0.3)
    plt.title("Aggregate Posterior")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")

    plt.subplot(1,3,2)
    plt.scatter(z_prior_2d[:,0], z_prior_2d[:,1], s=5, alpha=0.3)
    plt.title("VAE Prior")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")

    plt.subplot(1,3,3)
    plt.scatter(z_ddpm_2d[:,0], z_ddpm_2d[:,1], s=5, alpha=0.3)
    plt.title("Latent DDPM")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")

    plt.suptitle("Latent Distribution Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("Project1/plot.png")