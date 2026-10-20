# Assignment 1: Convolutional Variational Autoencoder
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a Convolutional Variational Autoencoder trained using the MNIST Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
Purpose: This class builds the framework for the Convolutional Variational Autoencoder
Parameters: None
Pre: None
Post: None
Return: ConvVAE(), an object of the Convolutional Variational Autoencoder
Raise: None
"""
class ConvVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.feature_layer = nn.Conv2d(64, 64, kernel_size=2, stride=1)

        self.fc_mu = nn.Conv2d(64, latent_channels, kernel_size=1)

        self.fc_logvar = nn.Conv2d(64, latent_channels, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    """
    Purpose: Simulates the encoding step of the Autoencoder
    Parameters: x - the images being processed
    Pre: x - the images being processed
    Post: x - the features are extracted
    Return: mu, logvar - the mean and log variance
    Raise: None
    """
    def encode(self, x):
        h = self.encoder(x)
        h = F.relu(self.feature_layer(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    """
    Purpose:  Simulates the reparameterization step of the Autoencoder
    Parameters: mu, logvar - the mean and log variance
    Pre: mu, logvar - the mean and log variance
    Post: mu, logvar - the two values are used to create a latent sample
    Return: lat_sample - the latent sample used to pass to the decoder
    Raise: None
    """
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        lat_sample = mu + eps * std
        return lat_sample

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
    Return: reconstruction, mu, logvar - the reconstruction value, the mean, and log variance
    Raise: None
    """
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

"""
Purpose: Download and ready the MNIST Dataset for training and testing
Parameters: None
Pre: None
Post: The MNIST Dataset for training and testing becomes ready
Return: train_loader, test_loader - a tuple pair containing the corresponding images
Raise: None
"""
def load_dataset():
    print("==========MNIST Training and Test Data Loading...==========")
    transform = transforms.ToTensor()

    # Download and ready the image dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("==========MNIST Training and Test Data Successfully Loaded!==========")
    return train_dataset, train_loader, test_loader

"""
Purpose: Measures the reconstruction loss when the data was being passed through
Parameters: reconstruction - the reconstructed pixel value, images - the original image data,
mu - the mean, logvar - the log variance, beta - the value used to determine reconstruction priority
Pre: reconstruction, images, mu, logvar, beta - all values must be valid
Post: reconstruction, images, mu, logvar, beta - used to calculate loss
Return: loss, recon_loss, kl_loss - the loss values of the reconstruction
Raise: None
"""
def vae_loss(reconstruction, images, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(reconstruction, images, reduction="sum")

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss

"""
Purpose: Evaluates the Peak Signal to Noise Ratio (PSNR)
Parameters: model - the Autoencoder, data_loader - the image data, device - the device running the model
Pre: model, data_loader, device - All inputs must be valid
Post: model, data_loader, device - Used to evaluate the Peak Signal to Noise Ratio (PSNR)
Return: avg_psnr, avg_mse - The average Peak Signal to Noise Ratio and average Mean Squared Error
Raise: None
"""
def evaluate(model, data_loader, device):
    model.eval()

    total_psnr = 0.0
    total_mse = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            reconstruction, mu, logvar = model(images)

            mse = ((images - reconstruction).pow(2).flatten(start_dim=1).mean(dim=1))

            psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))

            total_psnr += psnr.sum().item()
            total_mse += mse.sum().item()
            total_images += images.size(0)

    avg_psnr = total_psnr / total_images
    avg_mse = total_mse / total_images

    return avg_psnr, avg_mse

"""
Purpose: Train the model and test it to check its Peak Signal to Noise Ratio
Parameter: None
Pre: None
Post: The model and its results after training and testing will be posted
Return: None
Raise: None
"""
def main():
    print("**********Start of Program**********")
    # Load the MNIST Datasets
    train_dataset, train_loader, test_loader = load_dataset()

    # Find device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Compile and set the model
    model = ConvVAE(latent_channels=4).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    print("==========Model Training==========")
    num_epochs = 80
    beta = 0.01

    # Train the Model
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        # Use only the images in the training loader to train the model (the labels are not needed)
        for images, _ in train_loader:

            # Moves images to device
            images = images.to(device)

            optimizer.zero_grad()

            # Forward Pass: Grab the Reconstructed Pixel, Mean, and Log Variance
            reconstruction, mu, logvar = model(images)

            # Calculate Loss
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mu, logvar, beta
            )

            # Backward Pass: Calculate Gradients and Update Model Weights
            loss.backward()
            optimizer.step()

            # Save current training progress
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        # Show the training data after each epoch
        n = len(train_dataset)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {total_loss/n:.4f} "
            f"Recon: {total_recon/n:.4f} "
            f"KL: {total_kl/n:.4f}"
        )

        # Check the PSNR after each epoch
        psnr, mse = evaluate(model, train_loader, device)

        print(
            f"Train PSNR: {psnr:.2f} dB | "
            f"Train MSE: {mse:.6f}"
        )

    print("==========Model Evaluation==========")
    # Evaluate the model after training
    psnr, mse = evaluate(model, test_loader, device)

    print(f"Test MSE:  {mse:.6f}")
    print(f"Test PSNR: {psnr:.2f} dB")

    print("**********End of Program**********")

if __name__ == "__main__":
    main()

