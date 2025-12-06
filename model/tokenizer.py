class CharTokenizer:
    def __init__(self, texts):
        chars = sorted(list(set("".join(texts))))
        self.stoi = {c: i+1 for i, c in enumerate(chars)}
        self.stoi["<PAD>"] = 0
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != 0)