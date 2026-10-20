# Assignment 1: Sequence to Token Autoregressive Model
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a Sequence to Token Autoregressive Model trained using the MNIST Dataset

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

"""
Purpose: Converts a binary image into a series of integer tokens
Parameters: image - the binary image being converted
Pre: image - must be a valid binary image
Post: image - used to create a series of integer tokens
Return: tokens.astype(np.int64) - the integer token representation of the binary image
Raise: None
"""
def image_to_tokens(image):
    bits = image.reshape(-1).cpu().numpy().astype(np.uint8)
    groups = bits.reshape(-1, 8)

    powers = 2 ** np.arange(8 - 1, -1, -1)
    tokens = (groups * powers).sum(axis=1)

    return tokens.astype(np.int64)

"""
Purpose: Converts the series of integer tokens into a binary image
Parameters: sequence - the series of integer token
Pre: sequence - must be a valid integer token series
Post: sequence - used to create a binary image
Return: bits.reshape(28, 28) - the binary image
Raise: None
"""
def tokens_to_image(sequence):
    sequence = np.asarray(sequence, dtype=np.uint8)

    bits = np.unpackbits(sequence[:, None], axis=1)
    return bits.reshape(28, 28)

"""
Purpose: Create the framework for the Transformer Neural Network that predicts next image tokens
Parameters: None
Pre: None
Post: None
Return: SeqToToken() - the object representing the Transformer
Raise: None
"""
class SeqToToken(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            257, 64
        )
        self.position_embedding = nn.Embedding(
            98, 64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="relu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.output_layer = nn.Linear(64, 256)

    """
    Purpose: Simulates the data fed through the model to predict tokens
    Parameters: input_ids - the data being fed through the network
    Pre: input_ids - must be valid
    Post: input_ids - used to produce logits
    Return: self.output_layer(x) - the logits output that predicts the next token
    Raise: None
    """
    def forward(self, input_ids):
        batch_size, length = input_ids.shape

        positions = torch.arange(
            length, device=input_ids.device
        ).unsqueeze(0)

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        causal_mask = torch.triu(
            torch.ones(
                length, length,
                device=input_ids.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        return self.output_layer(x)

"""
Purpose: Evaluates the Transformer by generating new image token sequences
Parameters: model - the Transformer, device - the device used to run, temperature - 
Pre: model - must be created
Post: Creates a NumPy array of integers used for converting back into an image
Return: generated[0, 1:].cpu().numpy() - the NumPy array of integers used to convert back into an image
Raise: None
"""
@torch.no_grad()
def generate_sequence(model, device, temperature=0.8):
    model.eval()

    generated = torch.tensor(
        [[256]], dtype=torch.long, device=device
    )

    for _ in range(98):
        logits = model(generated)
        next_logits = logits[0, -1] / max(temperature, 1e-6)

        probabilities = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities, num_samples=1
        ).view(1, 1)

        generated = torch.cat(
            [generated, next_token], dim=1
        )

    return generated[0, 1:].cpu().numpy()

"""
Purpose: Simulates the training and testing of the Transformer for creating Sequence to Token predictions
Parameters: None
Pre: None
Post: A Transformer Model is trained and tested for Sequence to Token Predictions
Return: None
Raise: None
"""
def main():
    print("**********Start of Program**********")
    # Generate Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Gather Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    print("==========MNIST Training and Test Data Loading...==========")
    # Gather MNIST Dataset
    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    images = torch.stack([dataset[i][0] for i in range(100)])

    binary_images = (images >= (127 / 255)).to(torch.uint8)

    # Transform Images to Tokens
    tokens_np = np.stack([
        image_to_tokens(image) for image in binary_images
    ])

    tokens = torch.tensor(tokens_np, dtype=torch.long)

    assert tokens.shape == (100, 98)
    assert tokens.min().item() >= 0
    assert tokens.max().item() < 256

    print("Token tensor shape:", tuple(tokens.shape))
    print("Sequence length:", 98)
    print("Token vocabulary:", 256)

    bos_column = torch.full(
        (100, 1), 256, dtype=torch.long
    )

    model_inputs = torch.cat([bos_column, tokens[:, :-1]], dim=1)
    targets = tokens

    loader = DataLoader(
        TensorDataset(model_inputs, targets),
        batch_size=10,
        shuffle=True
    )

    print("==========MNIST Training and Test Data Successfully Loaded!==========")

    print("==========Model Training Start!==========")
    # Compile and set the model
    model = SeqToToken().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    # Train the Model
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_inputs)

            loss = criterion(
                logits.reshape(-1, 256),
                batch_targets.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs}, "
                f"loss={total_loss / len(loader):.4f}"
            )

    print("==========Model Evaluation Start!==========")
    torch.save(model.state_dict(), "sequence_transformer.pt")
    training_set = {
        tuple(sequence.tolist()) for sequence in tokens_np
    }

    NUM_GENERATED = 100

    generated_np = np.stack([
        generate_sequence(model, device) for _ in range(NUM_GENERATED)
    ])

    matches = sum(
        tuple(sequence.tolist()) in training_set
        for sequence in generated_np
    )

    match_rate = matches / NUM_GENERATED

    generated_images = np.stack([
        tokens_to_image(sequence)
        for sequence in generated_np
    ])

    # Save Training and Generated Tokens
    np.save("training_tokens.npy", tokens_np)
    np.save("generated_tokens.npy", generated_np)
    np.save("generated_images.npy", generated_images)

    # Display Results
    results = {
        "num_training_images": 100,
        "sequence_length": 98,
        "vocabulary_size": 256,
        "num_generated": NUM_GENERATED,
        "exact_training_matches": int(matches),
        "exact_match_rate": float(match_rate),
    }

    with open("evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    print("**********End of Program**********")

if __name__ == "__main__":
    main()
