import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricAttentionBias(nn.Module):
    def __init__(self, d1, d2, d3, H, d_model, d34096):
        super().__init__()

        self.d1 = d1  # TODO hyperparameter
        self.d2 = d2  # TODO hyperparameter
        self.d3 = d3  # TODO hyperparameter
        self.H = H  # TODO hyperparameter
        self.d_model = d_model  # TODO hyperparameter

        self.d1flattening = nn.Linear(d_model, d1)
        self.d1d2 = nn.Linear(64 * d1, d2)
        self.d2Hd3 = nn.Linear(d2, H * d3)

        # shared geometric template bank, shape [4096, d3]
        self.d34096 = d34096

        self.d2_norm = nn.LayerNorm(d2)
        self.Hd3_norm = nn.LayerNorm(H * d3)

    def forward(self, x):
        # x -> B, 64, d_model
        B, T, C = x.shape
        assert T == 64

        # compress d_model to d1 so the whole-board representation isnt enormous
        logits = self.d1flattening(x)  # B, 64, d_model --> B, 64, d1

        # flatten into one vector so GAB can look at the entire board
        logits = logits.reshape(B, 64 * self.d1)  # B, 64, d1 --> B, 64*d1

        # transform concatenated square representations into one global board representation
        logits = self.d1d2(logits)  # B, 64*d1 --> B, d2

        # add non-linearity and LayerNorm
        logits = F.gelu(logits)
        logits = self.d2_norm(logits)

        # transform d2 into d3 coefficients for each of the H attention heads
        logits = self.d2Hd3(logits)  # B, d2 --> B, H*d3

        # add non-linearity and LayerNorm
        logits = F.gelu(logits)
        logits = self.Hd3_norm(logits)

        # separate the H sets of d3 coefficients
        logits = logits.reshape(B, self.H, self.d3)  # B, H*d3 --> B, H, d3

        # transform each heads d3 coefficients into all 4096 square-pair biases
        logits = torch.einsum("bhi,oi->bho", logits, self.d34096)  # B, H, d3 --> B, H, 4096

        # reshape 4096 square pairs into a 64x64 attention bias matrix
        logits = logits.reshape(B, self.H, 64, 64)  # B, H, 4096 --> B, H, 64, 64

        return logits
