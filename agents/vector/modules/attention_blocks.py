import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


def get_causal_mask(seq):
    """For masking out the subsequent info."""
    batch_size, batch_length = seq.shape[:2]
    causal_mask = (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=seq.device), diagonal=1)).bool()
    return causal_mask


def get_causal_mask_with_batch_length(batch_length, device):
    """For masking out the subsequent info."""
    causal_mask = (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=device), diagonal=1)).bool()
    return causal_mask


def get_vector_mask(batch_length, device):
    mask = torch.ones((1, 1, batch_length), device=device).bool()
    # mask = torch.ones((1, batch_length, 1), device=device).bool()
    return mask


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # Get the appropriate negative infinity value for the data type of attn, FP32/FP16/BF16
            neg_inf = -torch.finfo(attn.dtype).max
            attn = attn.masked_fill(mask == 0, neg_inf)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.silu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            num_heads, feat_dim, feat_dim // num_heads, feat_dim // num_heads, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class AttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            num_heads, feat_dim, feat_dim // num_heads, feat_dim // num_heads, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask=None):
        output, attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def forward(self, feat):
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = einops.repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = einops.repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, position : position + 1, :]
        return feat


class PositionalEncoding2D(nn.Module):
    def __init__(self, max_length: int, num_objects: int, embed_dim: int):
        super().__init__()
        self.max_length = max_length
        self.num_objects = num_objects
        self.embed_dim = embed_dim

        self.spatial_emb = nn.Embedding(self.num_objects, embed_dim)
        self.temporal_emb = nn.Embedding(self.max_length, embed_dim)

    def get_pos_emb(self, device, batch_size):
        spatial_emb = self.spatial_emb(torch.arange(self.num_objects, device=device))
        spatial_emb = einops.rearrange(spatial_emb, "Obj D -> 1 Obj D")
        temporal_emb = self.temporal_emb(torch.arange(self.max_length, device=device))
        temporal_emb = einops.rearrange(temporal_emb, "L D -> L 1 D")

        pos_emb = spatial_emb + temporal_emb  # [1, Obj, D] + [L, 1, D] = [L, Obj, D], broadcast
        pos_emb = einops.repeat(pos_emb, "L Obj D -> B L Obj D", B=batch_size)
        return pos_emb

    def forward(self, feat):
        pos_emb = self.get_pos_emb(feat.device, feat.shape[0])
        feat = feat + pos_emb[:, : feat.shape[1]]

        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1

        pos_emb = self.get_pos_emb(feat.device, feat.shape[0])
        feat = feat + pos_emb[:, position : position + 1]

        return feat
