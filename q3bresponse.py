# Assignment 1: Sequence to Sequence Autoregressive Model
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a Sequence to Sequence Autoregressive Model trained using the MNIST Dataset

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

    weights = 2 ** np.arange(8 - 1, -1, -1)
    unsigned_values = (groups * weights).sum(axis=1)

    signed_values = np.where(
        unsigned_values >= 128,
        unsigned_values - 256,
        unsigned_values
    )

    token_ids = signed_values + 128
    return token_ids.astype(np.int64)

"""
Purpose: Converts the series of integer tokens into a binary image
Parameters: sequence - the series of integer token
Pre: sequence - must be a valid integer token series
Post: sequence - used to create a binary image
Return: bits.reshape(28, 28) - the binary image
Raise: None
"""
def tokens_to_image(token_ids):
    signed_values = np.asarray(token_ids, dtype=np.int64) - 128

    unsigned_values = np.mod(signed_values, 256).astype(np.uint8)

    bits = np.unpackbits(unsigned_values[:, None], axis=1)
    return bits.reshape(28, 28)

"""
Purpose: Create the framework for the Transformer Neural Network that predicts an output token from a source token
Parameters: None
Pre: None
Post: None
Return: SeqToSeq() - the object representing the Transformer
Raise: None
"""
class SeqToSeq(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            258, 128, padding_idx=25
        )
        self.position_embedding = nn.Embedding(98, 128)

        self.transformer = nn.Transformer(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )

        self.output = nn.Linear(128, 256)

    """
    Purpose: Converts tokens into vectors containing identity and position
    Parameters: ids - the token IDs
    Pre: ids - must be valid IDs
    Post: ids - used to create vectors
    Return: self.token_embedding(ids) + self.position_embedding(positions) - the vector containing identity and position
    Raise: None
    """
    def embed(self, ids):
        positions = torch.arange(
            ids.size(1), device=ids.device
        ).unsqueeze(0)
        return (
            self.token_embedding(ids)
            + self.position_embedding(positions)
        )

    """
    Purpose: Simulates the data fed through the model to predict next tokens
    Parameters: src - The details about the eq, size, and device, tgt_in - the target
    Pre: src, tgt_in -  must both be valid
    Post: src, tgt_in - used to create logits
    Return: self.output(output) - the logits output that predicts the next token
    Raise: None
    """
    def forward(self, src, tgt_in):
        src_padding_mask = src.eq(25)
        tgt_padding_mask = tgt_in.eq(25)

        tgt_len = tgt_in.size(1)
        causal_mask = torch.triu(
            torch.ones(
                tgt_len, tgt_len,
                dtype=torch.bool,
                device=tgt_in.device
            ),
            diagonal=1
        )

        output = self.transformer(
            src=self.embed(src),
            tgt=self.embed(tgt_in),
            tgt_mask=causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.output(output)

"""
Purpose: Evaluates the Transformer by generating new image token sequences
Parameters: model - the Transformer, device - the device used to run, temperature - 
Pre: model - must be created
Post: Creates a NumPy array of integers used for converting back into an image
Return: decoder_ids[0, 1:].cpu().numpy() - the NumPy array of integers used to convert back into an image
Raise: None
"""
@torch.no_grad()
def generate_from_source(source, model, device, temperature=0.8):
    model.eval()
    source = torch.as_tensor(
        source, dtype=torch.long, device=device
    ).reshape(1, -1)

    decoder_ids = torch.tensor(
        [[256]], dtype=torch.long, device=device
    )

    for _ in range(98):
        logits = model(source, decoder_ids)
        next_logits = logits[0, -1] / max(temperature, 1e-6)

        # Prevent invalid special tokens from being emitted.
        probabilities = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(
            probabilities, num_samples=1
        ).view(1, 1)

        decoder_ids = torch.cat([decoder_ids, next_id], dim=1)

    return decoder_ids[0, 1:].cpu().numpy()

"""
Purpose: Simulates the training and testing of the Transformer for creating Sequence to Sequence predictions
Parameters: None
Pre: None
Post: A Transformer Model is trained and tested for Sequence to Sequence Predictions
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("==========MNIST Training and Test Data Loading...==========")
    # Gather MNIST Dataset
    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # Transform Images to Tokens
    images = torch.stack([dataset[i][0] for i in range(100)])
    binary_images = (images >= (127 / 255)).to(torch.uint8)

    tokens_np = np.stack([image_to_tokens(img) for img in binary_images])
    tokens = torch.tensor(tokens_np, dtype=torch.long)

    assert tokens.shape == (100, 98)
    assert tokens.min() >= 0 and tokens.max() < 256

    bos = torch.full((100, 1), 256, dtype=torch.long)

    decoder_inputs = torch.cat([bos, tokens[:, :-1]], dim=1)

    loader = DataLoader(
        TensorDataset(tokens, decoder_inputs, tokens),
        batch_size=10,
        shuffle=True
    )

    print("==========MNIST Training and Test Data Successfully Loaded!==========")

    print("==========Model Training Start!==========")
    # Compile and Set the Model
    model = SeqToSeq().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=25)

    # Train the Model
    num_epochs = 300

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for src, tgt_in, tgt_out in loader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt_in)

            loss = loss_fn(
                logits.reshape(-1, 256),
                tgt_out.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"loss={total_loss / len(loader):.4f}"
            )

    # Save Masks
    torch.save(model.state_dict(), "seq2seq_masked.pt")

    training_set = {tuple(row.tolist()) for row in tokens_np}

    generated = np.stack([
        generate_from_source(tokens_np[i % 100], model, device)
        for i in range(100)
    ])

    matches = sum(
        tuple(row.tolist()) in training_set for row in generated
    )

    print("==========Model Evaluation Start!==========")

    results = {
        "num_training_images": 100,
        "sequence_length": 98,
        "vocabulary_size": 256,
        "num_generated": len(generated),
        "exact_training_matches": int(matches),
        "exact_match_rate": matches / len(generated),
    }

    # Save Generated Tokens
    np.save("seq2seq_generated_tokens.npy", generated)
    np.save(
        "seq2seq_generated_images.npy",
        np.stack([tokens_to_image(row) for row in generated])
    )

    with open("seq2seq_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    print("**********End of Program**********")

if __name__ == "__main__":
    main()
