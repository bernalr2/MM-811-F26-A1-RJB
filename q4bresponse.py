# Assignment 1: Image Generation Convolutional Variational Autoencoder
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a Convolutional Variational Autoencoder trained using the CIFAR-10 dataset
# This is for continuous based Image Generation

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from torchmetrics.image.fid import FrechetInceptionDistance

num_epochs = 10

AE_PATH = "continuous_ae_best.pt"
PRIOR_PATH = "latent_transformer.pt"
OUTPUT_DIR = "generated_images"

torch.manual_seed(42)

"""
Purpose: This class builds the framework for the Convolutional Variational Autoencoder
Parameters: None
Pre: None
Post: None
Return: ConvVAE(), an object of the Convolutional Variational Autoencoder
Raise: None
"""
class ConvVAE(nn.Module):
    def __init__(self, latent_channels=5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1),
        )

    """
    Purpose: Simulates the encoding step of the Autoencoder
    Parameters: x - the images being processed
    Pre: x - the images being processed
    Post: x - the features are extracted
    Return: self.encoder(x) - the extracted features
    Raise: None
    """
    def encode(self, x):
        return self.encoder(x)

    """
    Purpose: Simulates the decoder step of the Autoencoder
    Parameters: z - the latent sample from the reparameterize step
    Pre: z - the latent sample from the reparameterize step
    Post: z - the latent sample is used to reconstruct into a pixel value
    Return: reconstruction - the reconstructed pixel value
    Raise: None
    """
    def decode(self, z):
        reconstruction = torch.sigmoid(self.decoder(z))
        return reconstruction

    """
    Purpose: Processes the data through the layers of the Convolutional Variational Autoencoder
    Parameters: x - the image data being passed
    Pre: x - the image data being passed
    Post: x - the image data being passed is reconstructed into a pixel value
    Return: reconstruction, z - the reconstruction value and the extracted features
    Raise: None
    """
    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z

"""
Purpose: Evaluate the ConvVAE on its reconstruction accuracy
Parameters: model - the trained ConvVAE model, loader - the loaded image data, device - the device being used to run
Pre: model - must be trained first before evaluating
Post: The evaluation will return results on how the reconstructions behaved
Return: global_mse, global_psnr, mean_image_psnr - the Mean Squared Error and the Peak Signal to Noise Ratio (PSNR)
Raise: None
"""
@torch.no_grad()
def evaluate_autoencoder(model, loader, device):
    model.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    fid.set_dtype(torch.float64)

    total_squared_error = 0.0
    total_pixels = 0
    psnr_sum = 0.0
    image_count = 0

    for images, _ in loader:
        images = images.to(device)
        recon, _ = model(images)

        squared_error = (images - recon).square()
        total_squared_error += squared_error.sum().item()
        total_pixels += images.numel()

        mse_each = squared_error.flatten(1).mean(1)
        psnr_each = 10 * torch.log10(1.0 / mse_each.clamp_min(1e-10)
                                     )
        psnr_sum += psnr_each.sum().item()
        image_count += images.size(0)

        fid.update(images, real=True)

        reconstruction, _ = model(images)
        reconstruction = reconstruction.clamp(0, 1)

        fid.update(reconstruction, real=False)

    global_mse = total_squared_error / total_pixels
    global_psnr = -10 * math.log10(max(global_mse, 1e-10))

    mean_image_psnr = psnr_sum / image_count
    fid_score = fid.compute().item()

    return global_mse, global_psnr, mean_image_psnr, fid_score

"""
Purpose: Trains the ConvVAE Model
Parameters: model - the newly created model, train_loader - the training data, val_loader - the value data, device
Pre: None
Post: model - will be trained
Return: model - the trained model
Raise: None
"""
def train_autoencoder(model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, z = model(images)
            loss = F.mse_loss(recon, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        global_mse, global_psnr, mean_image_psnr, fid_score = evaluate_autoencoder(model, val_loader, device)

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"Train MSE: {running_loss/count:.6f} | "
            f"PSNR: {global_psnr:.2f} dB | "
            f"FID: {fid_score:.2f}"
        )

    return model

"""
Purpose: Gathers statistics such as mean, variance, and std from the latent generations
Parameters: model - the ConvVAE being trained, loader - the data used, device - the device used to run the model
Pre: None
Post: The latent stats are computed
Return: mean.view(1, 1, -1), std.view(1, 1, -1) - the mean and std statistics
Raise: None
"""
@torch.no_grad()
def compute_latent_stats(model, loader, device):
    model.eval()
    channel_sum = torch.zeros(5, device=device)
    channel_sq_sum = torch.zeros_like(channel_sum)
    count = 0

    for images, _ in loader:
        images = images.to(device)
        z = model.encode(images)

        channel_sum += z.sum(dim=(0, 2, 3))
        channel_sq_sum += z.square().sum(dim=(0, 2, 3))
        count += z.size(0) * z.size(2) * z.size(3)

    mean = channel_sum / count
    variance = (channel_sq_sum / count - mean.square()).clamp_min(1e-8)
    std = variance.sqrt().clamp_min(1e-4)

    return mean.view(1, 1, -1), std.view(1, 1, -1)

"""
Purpose: Saves the latent stats to be referred to later
Parameters: model - the ConvVAE being trained, loader - the data used, device - the device used to run the model
Pre: None
Post: The mean and std are calculated
Return: mean, std - the mean and standard deviation
Raise: None
"""
@torch.no_grad()
def save_latent_stats(model, loader, device):
    mean, std = compute_latent_stats(model, loader, device)
    torch.save(
        {"mean": mean.cpu(), "std": std.cpu()},
        "latent_stats.pt",
    )
    return mean, std

"""
Purpose: Simulates a model that predicts a sequence of latent vectors to generate into an image
Parameters: None
Pre: None
Post: None
Return: ContinuousLatentTransformer - the object representing the model
Raise: None
"""
class CLatentTransformer(nn.Module):
    def __init__(
        self,
        latent_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_tokens=64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_tokens = max_tokens

        self.input_projection = nn.Linear(latent_dim, d_model)
        self.position = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, latent_dim)

        self.start_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

    """
    Purpose: Runs the data through the Continuous Latent Transformer
    Parameters: inputs - the input data
    Pre: inputs - must be valid
    Post: inputs - will be used to create the projection
    Return: self.output_projection - the projection output
    Raise: None
    """
    def forward(self, inputs):
        B, T, _ = inputs.shape
        x = self.input_projection(inputs)
        x = x + self.position[:, :T]

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        x = self.transformer(x, mask=causal_mask)
        return self.output_projection(self.norm(x))

    """
    Purpose: Samples from the data being passed through the encoder
    Parameters: batch_size - the number of image data being used, device - the device used the run the model, temperature
    Pre: None
    Post: Creates a sample from the encoder data
    Return: torch.cat(generated, dim=1) - the taken sample
    Raise: None
    """
    @torch.no_grad()
    def sample(self, batch_size, device, temperature=1.0):
        self.eval()
        tokens = self.start_token.expand(batch_size, -1, -1).to(device)
        generated = []

        for _ in range(self.max_tokens):
            prediction = self(tokens)[:, -1:, :]
            generated.append(prediction)

            next_token = prediction + (
                torch.randn_like(prediction) * temperature
            )
            tokens = torch.cat([tokens, next_token], dim=1)

        return torch.cat(generated, dim=1)

"""
Purpose: Trains the Transformer to predict next tokens based on previous tokens
Parameters: autoencoder - the ConvVAE, prior - parameters before starting, train_loader - the data being used, device
Pre: None
Post: The Transformer is trained
Return: None
Raise: None
"""
def train_transformer(
    autoencoder, prior, train_loader, mean, std, device
):
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(prior.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(num_epochs):
        prior.train()
        loss_sum = 0.0
        count = 0

        for images, _ in train_loader:
            images = images.to(device)

            with torch.no_grad():
                z = autoencoder.encode(images)
                z = z.permute(0, 2, 3, 1).reshape(
                    images.size(0), 64, 5
                )
                z = (z - mean) / std

            start = prior.start_token.expand(
                z.size(0), 1, 5
            )
            inputs = torch.cat([start, z[:, :-1]], dim=1)
            targets = z

            optimizer.zero_grad(set_to_none=True)
            prediction = prior(inputs)
            loss = F.mse_loss(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                prior.parameters(), max_norm=1.0
            )
            optimizer.step()

            loss_sum += loss.item() * images.size(0)
            count += images.size(0)

        print(
            f"Transformer: {epoch+1:03d}/{num_epochs} "
            f"MSE: {loss_sum/count:.6f}"
        )

        torch.save(prior.state_dict(), PRIOR_PATH)

"""
Purpose: Generates images using the data from the ConvVAE
Parameters: autoencoder - the ConvVAE, prior - parameters before starting, train_loader - the data being used, device
Pre: None
Post: Test images are generated
Return: None
Raise: None
"""
@torch.no_grad()
def generate_images(
    autoencoder, prior, mean, std, device,
    num_images=10000, batch_size=128, temperature=0.5
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    autoencoder.eval()
    prior.eval()

    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".png"):
            os.remove(os.path.join(OUTPUT_DIR, filename))

    written = 0
    while written < num_images:
        n = min(batch_size, num_images - written)

        z = prior.sample(
            n, device=device, temperature=temperature
        )

        z = z * std + mean
        z = z.reshape(n, 8, 8, 5)
        z = z.permute(0, 3, 1, 2).contiguous()

        images = autoencoder.decode(z).clamp(0, 1)

        for j in range(n):
            path = os.path.join(
                OUTPUT_DIR, f"{written+j:06d}.png"
            )
            utils.save_image(images[j], path)

        written += n

    print(f"Saved {written} generated images to {OUTPUT_DIR}")

"""
Purpose: Trains and Tests the ConvVAE using a Continuous Discrete Transformer
Parameters: None
Pre: None
Post: A ConvVAE is trained and tested and images are generated
Return: None
Raise: None
"""
def main():
    print("**********Start of Program**********")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("==========CIFAR10 Training and Test Data Loading...==========")
    # Load CIFAR10 Dataset
    transform = transforms.ToTensor()
    full_train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform
    )

    train_size = 45000
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    print("==========CIFAR10 Training and Test Data Successfully Loaded!==========")

    print("==========Model Training Start!==========")

    autoencoder = ConvVAE(5).to(device)

    autoencoder = train_autoencoder(autoencoder, train_loader, val_loader, device)

    print("==========Model Evaluation Start!==========")
    test_metrics = evaluate_autoencoder(autoencoder, test_loader, device)

    mean, std = save_latent_stats(autoencoder, train_loader, device)

    prior = CLatentTransformer(latent_dim=5, d_model=256, n_heads=8, n_layers=6, max_tokens=64).to(device)

    print("==========Transformer Training Start!==========")
    train_transformer(autoencoder, prior, train_loader, mean.to(device), std.to(device), device)

    print("==========Images Generating...==========")
    generate_images(autoencoder, prior, mean.to(device), std.to(device), device, num_images=10000, temperature=0.5)

    print("**********End of Program**********")


if __name__ == "__main__":
    main()
