import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        max_len: int = 100,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input_seq):
        # input_seq: (B, L)
        B, L = input_seq.size()
        pos = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, -1)

        x = self.item_emb(input_seq) + self.pos_emb(pos)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # causal mask: no attending to the future
        mask = torch.triu(torch.ones(L, L, device=input_seq.device), diagonal=1).bool()
        x = self.encoder(x, mask)
        return x

    def predict_next(self, input_seq):
        # input_seq: (B, L)
        h = self.forward(input_seq)          # (B, L, d)
        last_hidden = h[:, -1, :]           # (B, d)
        last_hidden = self.output_layer(last_hidden)  # (B, d)
        item_emb = self.item_emb.weight     # (num_items+1, d)
        logits = last_hidden @ item_emb.t() # (B, num_items+1)
        return logits
