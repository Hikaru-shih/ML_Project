import json
import torch

from tokenizer import CharTokenizer
from transformer import TinyTransformer

MODEL_PATH = "../results/model.pt"
TOKENIZER_PATH = "../results/tokenizer.json"
BLOCK_SIZE = 256
MAX_NEW = 120


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=120, temperature=0.8):
    model.eval()

    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    eos_token = tokenizer.encode("<EOS>")[0]  # char-level means the first "<"

    for _ in range(max_new):
        ids_cond = ids[:, -model.block_size:]
        logits = model(ids_cond)[0, -1] / temperature

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # STOP CONDITION:
        # If model outputs '<' (the start of "<EOS>"), we assume EOS begins.
        if next_id.item() == eos_token:
            break

        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)

    return tokenizer.decode(ids[0].tolist())


def main():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        stoi = json.load(f)

    tokenizer = CharTokenizer([])
    tokenizer.stoi = stoi
    tokenizer.itos = {i: c for c, i in stoi.items()}

    vocab_size = len(stoi)

    model = TinyTransformer(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    world_state = "IN:Location: forest; Time: night; Player: injured; NPC: hunter; Weather: foggy\nOUT:"

    output = generate(model, tokenizer, world_state)
    print("Generated Event:")
    print(output)


if __name__ == "__main__":
    main()