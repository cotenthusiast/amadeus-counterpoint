import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricAttentionBias(nn.Module):
    ...

# Vectorized Multi-head attention 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.keys = nn.Linear(d_model, num_heads * self.head_size)
        self.queries = nn.Linear(d_model, num_heads * self.head_size)
        self.values = nn.Linear(d_model, num_heads * self.head_size)
        self.proj = nn.Linear(num_heads * self.head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        H = self.num_heads
        D = self.head_size

        k = self.keys(x) # B, T, C
        q = self.queries(x) 
        v = self.values(x) 

        k = k.reshape(B, T, H, D).transpose(1, 2) # B, T, C  --->  B, T, H, D  --->  B, H, T, D
        q = q.reshape(B, T, H, D).transpose(1, 2)
        v = v.reshape(B, T, H, D).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # B, H, T, D  @  (B, H, T, D).H  --->  B, H, T, T

        # add GAB bias to logits

        wei = F.softmax(wei, dim=-1) # B, H, T, T
        wei = self.dropout(wei) # B, H, T, T

        out = wei @ v # B, H, T, T  @  B, H, T, D  --->  B, H, T, D

        return self.proj(
                (out.transpose(1, 2)).reshape(B, T, C) # Switching H and T, reshaping the matrix from B, T, H, D to B, T, C, carrying out the learned projection transformation then doing dropout
            )
        