import torch
import torch.nn as nn
import torch.nn.functional as F

import chess
import math


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
