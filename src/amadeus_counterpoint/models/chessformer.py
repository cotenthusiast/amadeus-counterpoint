import torch
import torch.nn as nn

from amadeus_counterpoint.models.transformer_block import TransformerBlock
from amadeus_counterpoint.models.heads import PolicyHead, ValueHead


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
        nn.init.xavier_normal_(self.elo_low.weight)
        nn.init.xavier_normal_(self.elo_high.weight)

        # shared GAB template bank, one set of weights across every block
        self.d34096 = nn.Parameter(torch.empty(4096, d3))
        nn.init.xavier_normal_(self.d34096)

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

        # official endpoint weighting: weight_low grows with elo and
        # multiplies elo_low, weight_high shrinks with elo and multiplies
        # elo_high -- counterintuitive naming, reproduced exactly
        weight_low = elo / 5000
        weight_high = 1 - weight_low

        low = self.elo_low.weight[0]
        high = self.elo_high.weight[0]

        return (
            weight_low.unsqueeze(-1) * low
            + weight_high.unsqueeze(-1) * high
        )