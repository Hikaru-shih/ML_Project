import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

from tokenizer import CharTokenizer
from transformer import TinyTransformer

DATA_PATH = "../data/dataset.json"
RESULT_DIR = "../results/"
MODEL_PATH = "../results/model.pt"
TOKENIZER_PATH = "../results/tokenizer.json"

BLOCK_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 250
LR = 3e-4


# ----------------------------
# Dataset
# ----------------------------
class EventDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []

        for pair in data:
            text = f"IN:{pair['input']}\nOUT:{pair['output']}<EOS>"
            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.samples[idx])
        ids = ids[:BLOCK_SIZE]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    max_len = min(max_len, BLOCK_SIZE)

    padded = []
    targets = []

    for seq in batch:
        seq = seq[:max_len]
        pad_len = max_len - len(seq)

        seq_padded = torch.cat([
            seq,
            torch.zeros(pad_len, dtype=torch.long)
        ])

        target = torch.cat([
            seq[1:],
            torch.zeros(1, dtype=torch.long)
        ])
        target = torch.cat([
            target,
            torch.zeros(pad_len, dtype=torch.long)
        ])

        padded.append(seq_padded)
        targets.append(target)

    return torch.stack(padded), torch.stack(targets)


# ----------------------------
# Training
# ----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    all_texts = []
    for pair in raw_data:
        full_sample = f"IN:{pair['input']}\nOUT:{pair['output']}<EOS>"
        all_texts.append(full_sample)

    tokenizer = CharTokenizer(all_texts)

    dataset = EventDataset(raw_data, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    vocab_size = len(tokenizer.stoi)
    model = TinyTransformer(vocab_size=vocab_size, block_size=BLOCK_SIZE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch, target in loader:
            batch = batch.to(device)
            target = target.to(device)

            logits = model(batch)

            logits = logits.reshape(-1, vocab_size)
            target = target.reshape(-1)

            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        losses.append(avg)
        print(f"[Epoch {epoch+1}] Loss = {avg:.4f}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(RESULT_DIR, "loss_curve.png"))
    plt.close()

    torch.save(model.state_dict(), MODEL_PATH)

    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenizer.stoi, f)

    print("Training completed.")
    print("Model saved to", MODEL_PATH)
    print("Tokenizer saved to", TOKENIZER_PATH)


if __name__ == "__main__":
    train()