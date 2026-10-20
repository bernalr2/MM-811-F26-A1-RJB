# Assignment 1: Image Generation Vector Quantizer Variational Autoencoder
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a Vector Quantizer Variational Autoencoder trained using the CIFAR-10 dataset
# This is for diffusion based Image Generation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

"""
Purpose: Creates the Vector Quantizer layer used for the VQ-VAE
Parameters: None
Pre: None
Post: None
Return: VectorQuantizer(), the Vector Quantizer layer used for the VQ-VAE
Raise: None
"""
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=512, embedding_dim=10, commitment_beta=0.25):
        super().__init__()

        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta

        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    """
    Purpose: Runs the input data through the VQ
    Parameters: z -
    Pre: z - has to be a valid data input
    Post: z - is used to calculate the quantized_st, indices, and vq_loss
    Return: quantized_st, indices, vq_loss - the 
    Raise: None
    """
    def forward(self, z):
        B, D, H, W = z.shape

        flat_z = z.permute(0, 2, 3, 1).contiguous()
        flat_z = flat_z.view(-1, D)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * flat_z @ self.embedding.weight.t()
        )

        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices)
        quantized = quantized.view(B, H, W, D)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, z.detach())

        commitment_loss = self.commitment_beta * F.mse_loss(z, quantized.detach())

        vq_loss = codebook_loss + commitment_loss

        quantized_st = z + (quantized - z).detach()

        indices = indices.view(B, H, W)

        return quantized_st, indices, vq_loss

"""
Purpose: This class builds the framework for the Vector Quantizer Variational Autoencoder
Parameters: None
Pre: None
Post: None
Return: VQVAE(), an object of the Vector Quantizer Variational Autoencoder
Raise: None
"""
class VQVAE(nn.Module):
    def __init__(self, embedding_dim=10, num_codes=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedding_dim, 3, padding=1),
        )

        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    """
        Purpose: Simulates the encoding step of the VQVAE
        Parameters: images - the images being processed
        Pre: images - the images being processed
        Post: images - the features are extracted
        Return: quantized, indices, vq_loss - 
        Raise: None
        """
    def encode(self, images):
        z = self.encoder(images)
        quantized, indices, vq_loss = self.quantizer(z)
        return quantized, indices, vq_loss

    """
    Purpose: Simulates the decoder step of the Autoencoder
    Parameters: quantized - the latent sample from the quantize step
    Pre: quantized - the latent sample from the quantize step
    Post: quantized - the latent sample is used to reconstruct into a pixel value
    Return: self.decoder(quantized) - the reconstructed pixel value
    Raise: None
    """
    def decode(self, quantized):
        return self.decoder(quantized)

    """
    Purpose: Processes the data through the layers of the Convolutional Variational Autoencoder
    Parameters: images - the image data being passed
    Pre: images - the image data being passed
    Post: images - the image data being passed is reconstructed into a pixel value
    Return: reconstruction, indices, vq_loss - the reconstruction value, the mean, and ...
    Raise: None
    """
    def forward(self, images):
        quantized, indices, vq_loss = self.encode(images)
        reconstruction = self.decode(quantized)

        return reconstruction, indices, vq_loss

"""
Purpose: Download and ready the CIFAR10 Dataset for training and testing
Parameters: None
Pre: None
Post: The CIFAR10 Dataset for training and testing becomes ready
Return: train_loader, test_loader - a tuple pair containing the corresponding images
Raise: None
"""
def load_dataset(device):
    print("==========CIFAR10 Training and Test Data Loading...==========")
    # Load CIFAR10 Dataset
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print("==========CIFAR10 Training and Test Data Successfully Loaded!==========")
    return train_loader, test_loader

"""
Purpose: Evaluate the VQVAE using the CIFAR10 dataset
Parameters: model - the VQVAE, loader - the test data being used, device - the device running the model
Pre: model - must be trained
Post: model - will be trained
Return: global_mse, global_psnr, active_code_count, model.quantizer.num_codes, fid_score - the results from testing
Raise: None
"""
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    squared_error_sum = 0.0
    pixel_count = 0
    image_psnr_sum = 0.0
    image_count = 0
    active_codes = torch.zeros(model.quantizer.num_codes, dtype=torch.bool)

    for images, _ in loader:
        images = images.to(device)

        reconstruction, indices, _ = model(images)

        # Update FID with real and reconstructed images.
        fid_metric.update(images, real=True)
        fid_metric.update(reconstruction.clamp(0, 1), real=False,)

        squared_error = (images - reconstruction).pow(2)

        # Dataset-wide pixel-weighted MSE.
        squared_error_sum += squared_error.sum().item()
        pixel_count += images.numel()

        # Mean PSNR per image.
        mse_per_image = squared_error.flatten(1).mean(1)
        psnr_per_image = 10.0 * torch.log10(1.0 / mse_per_image.clamp_min(1e-10))

        image_psnr_sum += psnr_per_image.sum().item()
        image_count += images.size(0)

        active_codes[indices.cpu().reshape(-1).unique()] = True

    global_mse = squared_error_sum / pixel_count
    global_psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(global_mse, 1e-10))).item()

    active_code_count = active_codes.sum().item()

    # Calculate FID after all batches have been processed.
    fid_score = fid_metric.compute().item()

    return {
        "global_mse": global_mse,
        "global_psnr_db": global_psnr,
        "active_codes": active_code_count,
        "codebook_size": model.quantizer.num_codes,
        "fid": fid_score,
    }

"""
Purpose: Train the model and test it to check its Peak Signal to Noise Ratio (PSNR) and FID Score
Parameter: None
Pre: None
Post: The model and its results after training and testing will be posted
Return: None
Raise: None
"""
def main():
    print("**********Start of Program**********")
    torch.manual_seed(42)

    # Find Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR10 Dataset
    train_loader, test_loader = load_dataset(device)

    # Compile and set the model
    model = VQVAE(embedding_dim=10, num_codes=512).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    print("==========Model Training==========")
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()

        loss_sum = 0.0
        recon_sum = 0.0
        vq_sum = 0.0
        count = 0

        # Use only the images in the training loader to train the model (the labels are not needed)
        for images, _ in train_loader:

            # Moves images to device
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward Pass: Grab the reconstructed pixel, indices, and vq_loss
            reconstruction, indices, vq_loss = model(images)

            # Calculate Loss
            recon_loss = F.mse_loss(reconstruction, images)
            loss = recon_loss + vq_loss

            # Backward Pass: Calculate Gradients and Update Model Weights
            loss.backward()
            optimizer.step()

            # Save current training progress
            batch_size = images.size(0)
            loss_sum += loss.item() * batch_size
            recon_sum += recon_loss.item() * batch_size
            vq_sum += vq_loss.item() * batch_size
            count += batch_size

        # Display Training Progress
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Loss {loss_sum / count:.5f} | "
            f"Recon {recon_sum / count:.5f} | "
            f"VQ {vq_sum / count:.5f}"
        )

    # Evaluate the model with the test data
    metrics = evaluate(model, test_loader, device)

    print(
        f"Epoch {epoch + 1:03d}/{num_epochs} | "
        f"Loss {loss_sum/count:.5f} | "
        f"Recon {recon_sum/count:.5f} | "
        f"VQ {vq_sum/count:.5f} | "
        f"PSNR {metrics['global_psnr_db']:.2f} dB | "
        f"FID {metrics['fid']:.2f}"
    )

    # Show Final Results
    print(f"Test MSE: {metrics['global_mse']:.6f}")
    print(f"Test PSNR: {metrics['global_psnr_db']:.2f} dB")
    print(f"Test FID: {metrics['fid']:.4f}")

    print("**********End of Program**********")

if __name__ == "__main__":
    main()
