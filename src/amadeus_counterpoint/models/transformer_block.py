import torch.nn as nn
import torch.nn.functional as F

from amadeus_counterpoint.models.mha import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout,
        d1,
        d2,
        d3,
        d34096,
        d_ff,
    ):
        super().__init__()

        self.d_model = d_model  # TODO hyperparameter
        self.num_heads = num_heads  # TODO hyperparameter
        self.d_ff = d_ff  # TODO hyperparameter

        # multi-head attention with GAB
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            d1=d1,
            d2=d2,
            d3=d3,
            d34096=d34096,
        )

        # feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        # post-norm LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout before each residual connection
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def mlp(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.attention(x)))
        x = self.norm2(x + self.dropout2(self.mlp(x)))

        return x
