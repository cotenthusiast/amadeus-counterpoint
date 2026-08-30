import torch
import torch.nn as nn
import torch.nn.functional as F

import chess
import math

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
        self.keys = nn.Linear(d_model, d_model)
        self.queries = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)

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

class PolicyHead(nn.Module):
    def __init__(self, d_model, head_hid_dim):
        super().__init__()

        self.d_model = d_model  # TODO hyperparameter
        self.head_hid_dim = head_hid_dim  # TODO hyperparameter

        # transform each square into source and destination representations
        self.proj_sq_from = nn.Linear(d_model, head_hid_dim, bias=False)
        self.proj_sq_to = nn.Linear(d_model, head_hid_dim, bias=False)

        # generate queen, rook, bishop, and knight promotion biases
        self.promo_bias_proj = nn.Linear(head_hid_dim, 4, bias=False)

    def forward(self, x):
        # x -> B, 64, d_model
        B, _, _ = x.shape

        from_logits = self.proj_sq_from(x)
        to_logits = self.proj_sq_to(x)

        # score every source square against every destination square
        pairwise_logits = (
            from_logits @ to_logits.transpose(1, 2)
        ) / math.sqrt(self.head_hid_dim)  # B, 64, 64

        logits = pairwise_logits.reshape(B, 64 * 64)  # B, 4096

        # generate Q/R/B/N biases for every rank-8 destination square
        rank7_indices = [chess.square(file, 6) for file in range(8)]
        rank8_indices = [chess.square(file, 7) for file in range(8)]

        rank8_features = to_logits[:, rank8_indices, :]  # B, 8, head_hid_dim
        promotion_biases = self.promo_bias_proj(rank8_features) * math.sqrt(self.head_hid_dim)  # B, 8, 4

        # combine rank7 -> rank8 base scores with each promotion-piece bias
        promotion_logits = []

        for from_file in range(8):
            from_sq = rank7_indices[from_file]

            for to_file in range(8):
                to_sq = rank8_indices[to_file]
                base_score = pairwise_logits[:, from_sq, to_sq]

                for piece_idx in range(4):
                    bias = promotion_biases[:, to_file, piece_idx]
                    promotion_logits.append((base_score + bias).unsqueeze(1))

        promotion_logits = torch.cat(promotion_logits, dim=1)  # B, 256
        logits = torch.cat([logits, promotion_logits], dim=1)  # B, 4352

        return logits

class ValueHead(nn.Module):
    def __init__(self, d_model, head_hid_dim):
        super().__init__()

        self.d_model = d_model  # TODO hyperparameter
        self.head_hid_dim = head_hid_dim  # TODO hyperparameter

        self.norm = nn.LayerNorm(d_model)
        self.fc_value_hid = nn.Linear(d_model, head_hid_dim)
        self.fc_value = nn.Linear(head_hid_dim, 3)

    def forward(self, x):
        # average all square representations into one board representation
        x = x.mean(dim=1)  # B, 64, d_model --> B, d_model

        # normalize the global board representation
        x = self.norm(x)

        # predict win, draw, and loss logits
        x = self.fc_value_hid(x)  # B, d_model --> B, head_hid_dim
        x = F.relu(x)
        x = self.fc_value(x)  # B, head_hid_dim --> B, 3

        return x

class Chessformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        dropout,
        d1,
        d2,
        d3,
        d_ff,
        head_hid_dim,
        input_dim,
        elo_dim
    ):
        super().__init__()

        self.elo_low = nn.Embedding(1, elo_dim)
        self.elo_high = nn.Embedding(1, elo_dim)        

        # shared GAB transformation
        self.d34096 = nn.Linear(d3, 4096, bias=False)

        # transform raw input features into model representations
        self.input_projection = nn.Linear(
            input_dim + 2 * elo_dim,
            d_model,
        )

        # repeated Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                d1=d1,
                d2=d2,
                d3=d3,
                d34096=self.d34096,
                d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])

        # final encoder normalization
        self.final_norm = nn.LayerNorm(d_model)

        # output heads
        self.policy_head = PolicyHead(d_model, head_hid_dim)
        self.value_head = ValueHead(d_model, head_hid_dim)

    def forward(self, x, player_elo, opponent_elo):

        # turn Elo numbers into learned skill representations
        player_emb = self.interpolate_elo(player_elo)      # B --> B, elo_dim
        opponent_emb = self.interpolate_elo(opponent_elo)  # B --> B, elo_dim

        # give the same global Elo context to every square
        player_emb = player_emb.unsqueeze(1).expand(-1, 64, -1)
        opponent_emb = opponent_emb.unsqueeze(1).expand(-1, 64, -1)

        # combine chess state and rating context
        x = torch.cat([x, player_emb, opponent_emb], dim=-1)

        # enter model space
        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def interpolate_elo(self, elo):

        # clamp into supported range
        elo = elo.clamp(0, 5000)

        alpha = elo / 5000

        low = self.elo_low.weight[0]
        high = self.elo_high.weight[0]

        return (
            (1 - alpha.unsqueeze(-1)) * low
            + alpha.unsqueeze(-1) * high
        )