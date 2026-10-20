# Assignment 1: Sequence to Histogram Autoregressive Model
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a Sequence to Histogram Autoregressive Model trained using the MNIST Dataset

import torch
import torch.nn as nn
import numpy as np
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

    return (signed_values + 128).astype(np.int64)

"""
Purpose: Create the framework for the Transformer Neural Network that predicts an output token from a source token
Parameters: None
Pre: None
Post: None
Return: SeqToHist() - the object representing the Transformer
Raise: None
"""
class SeqToHist(nn.Module):
    def __init__(
        self,
        vocab_size=0, #EMBED_VOCAB_SIZE
        d_model=192,
        nhead=6,
        num_layers=4,
        ff_dim=768,
        max_len=0 #SEQ_LEN
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(d_model)
        self.histogram_head = nn.Linear(d_model, vocab_size)

    """
    Purpose: Simulates the data fed through the model to predict tokens
    Parameters: input_ids - the data being fed through the network
    Pre: input_ids - must be valid
    Post: input_ids - used to produce logits
    Return: self.histogram_head(self.norm(x)) - the logits output that predicts the next token
    Raise: None
    """
    def forward(self, input_ids):
        batch, length = input_ids.shape

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
                dtype=torch.bool,
                device=input_ids.device
            ),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        return self.histogram_head(self.norm(x))

"""
Purpose: Evaluates the Transformer by generating new image token sequences
Parameters: model - the Transformer, device - the device used to run, temperature - 
Pre: model - must be created
Post: Creates a NumPy array of integers used for converting back into an image
Return: generated[0, 1:].cpu().numpy() - the NumPy array of integers used to convert back into an image
Raise: None
"""
@torch.no_grad()
def generate_sequence(model, device, mode="greedy", temperature=1.0, seq_len=0, bos_id=0):
    model.eval()

    generated = torch.tensor(
        [[bos_id]], dtype=torch.long, device=device
    )

    for _ in range(seq_len):
        logits = model(generated)
        next_logits = logits[0, -1]

        if mode == "greedy":
            next_token = next_logits.argmax().view(1, 1)
        else:
            probabilities = torch.softmax(
                next_logits / max(temperature, 1e-6),
                dim=-1
            )
            next_token = torch.multinomial(
                probabilities, num_samples=1
            ).view(1, 1)

        generated = torch.cat(
            [generated, next_token], dim=1
        )

    return generated[0, 1:].cpu().numpy()

"""
Purpose: Simulates the training and testing of the Transformer for creating Sequence to Histogram predictions
Parameters: None
Pre: None
Post: A Transformer Model is trained and tested for Sequence to Histogram Predictions
Return: None
Raise: None
"""
def main():
    print("**********Start of Program**********")
    # Generate Seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Gather Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==========MNIST Training and Test Data Loading...==========")
    # Load the MNIST Dataset
    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # Transform Images to Tokens
    images = torch.stack([
        dataset[i][0] for i in range(512)
    ])

    binary_images = (images >= (127 / 255)).to(torch.uint8)

    tokens_np = np.stack([
        image_to_tokens(image) for image in binary_images
    ])

    NUM_IMAGES = len(tokens_np)
    SEQ_LEN = tokens_np.shape[1]
    VOCAB_SIZE = 256
    BOS_ID = 256
    EMBED_VOCAB_SIZE = 257

    tokens = torch.as_tensor(tokens_np, dtype=torch.long)
    bos = torch.full((NUM_IMAGES, 1), BOS_ID, dtype=torch.long)

    X = torch.cat([bos, tokens[:, :-1]], dim=1)

    Y = tokens

    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=32,
        shuffle=True
    )

    print("==========MNIST Training and Test Data Successfully Loaded!==========")

    print("==========Model Training Start!==========")

    # Compile and set model
    model = SeqToHist(vocab_size=EMBED_VOCAB_SIZE, max_len=SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()

    # Train the Model
    num_epochs = 300

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)

            loss = criterion(
                logits.reshape(-1, EMBED_VOCAB_SIZE),
                targets.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (
                    logits.argmax(dim=-1) == targets
            ).sum().item()
            total_tokens += targets.numel()

        mean_loss = total_loss / len(loader)
        token_accuracy = total_correct / total_tokens

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1:4d}/{num_epochs} | "
                f"loss={mean_loss:.5f} | "
                f"token_accuracy={token_accuracy:.4%}"
            )

    print("==========Model Evaluation Start!==========")
    model.eval()

    training_set = {tuple(row.tolist()) for row in tokens_np}
    NUM_GENERATED = 1000

    for mode in ["greedy", "sample"]:
        outputs = np.stack([
            generate_sequence(model=model, device=device, mode=mode, temperature=0.7, seq_len=SEQ_LEN, bos_id=BOS_ID)
            for _ in range(NUM_GENERATED)
        ])

        matches = sum(
            tuple(row.tolist()) in training_set
            for row in outputs
        )

        rate = matches / NUM_GENERATED

        print(f"Exact Matches: {matches}/{NUM_GENERATED}")
        print(f"Match Percentage: {rate:.2%}")

        # Save Outputs
        np.save(f"histogram_{mode}_generated.npy", outputs)

    print("**********End of Program**********")

if __name__ == "__main__":
    main()