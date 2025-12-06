import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, emb=128, nhead=4, depth=2, block_size=160):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, emb)
        self.pos_emb = nn.Embedding(block_size, emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.lm_head = nn.Linear(emb, vocab_size)

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        tokens = self.token_emb(x) + self.pos_emb(positions)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        out = self.transformer(tokens, mask)
        logits = self.lm_head(out)
        return logits
