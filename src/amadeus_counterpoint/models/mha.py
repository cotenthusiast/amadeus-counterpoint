import torch.nn as nn
import torch.nn.functional as F

from amadeus_counterpoint.models.gab import GeometricAttentionBias


class MultiHeadAttention(nn.Module):
    """Vectorized multi-head self-attention with geometric attention bias."""

    def __init__(self, d_model, num_heads, dropout, d1, d2, d3, d34096):
        super().__init__()

        # d_model must split evenly across all attention heads
        assert d_model % num_heads == 0

        self.d_model = d_model  # TODO hyperparameter
        self.num_heads = num_heads  # TODO hyperparameter
        self.head_size = d_model // num_heads  # TODO hyperparameter

        # project each square representation into queries, keys, and values
        self.keys = nn.Linear(d_model, d_model, bias=False)
        self.queries = nn.Linear(d_model, d_model, bias=False)
        self.values = nn.Linear(d_model, d_model, bias=False)

        # generate geometric attention biases for each head
        self.gab = GeometricAttentionBias(
            d1=d1,
            d2=d2,
            d3=d3,
            H=num_heads,
            d_model=d_model,
            d34096=d34096,
        )

        # recombine information from all attention heads
        self.proj = nn.Linear(d_model, d_model)

        # dropout applied to attention weights after softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B = batch size, T = 64 squares, C = d_model
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_size

        # project every square representation into Q, K, and V
        k = self.keys(x)  # B, T, C --> B, T, C
        q = self.queries(x)  # B, T, C --> B, T, C
        v = self.values(x)  # B, T, C --> B, T, C

        # split d_model across the H attention heads
        k = k.reshape(B, T, H, D).transpose(1, 2)  # B, T, C --> B, H, T, D
        q = q.reshape(B, T, H, D).transpose(1, 2)  # B, T, C --> B, H, T, D
        v = v.reshape(B, T, H, D).transpose(1, 2)  # B, T, C --> B, H, T, D

        # compare every square with every other square
        attn_logits = q @ k.transpose(-2, -1) * D**-0.5  # B, H, T, T

        # add one geometric 64x64 bias matrix per attention head
        attn_logits = attn_logits + self.gab(x)  # B, H, T, T

        # convert attention logits into attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # take a weighted combination of the value vectors
        out = attn_weights @ v  # B, H, T, T @ B, H, T, D --> B, H, T, D

        # recombine the H heads back into d_model
        out = out.transpose(1, 2).reshape(B, T, C)  # B, H, T, D --> B, T, C

        # mix information from the concatenated heads
        return self.proj(out)
