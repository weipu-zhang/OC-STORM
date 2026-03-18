import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .attention_blocks import get_vector_mask
from .attention_blocks import PositionalEncoding1D, PositionalEncoding2D, AttentionBlock, AttentionBlockKVCache


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dims, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        assert self.check_action_dims(action_dims), (
            "Currently only support Atari like action space = [A_dim] or Hollow Knight like action space = [2]*A_dim"
        )
        self.action_dims = action_dims
        self.action_choices = action_dims[0]  # only support same action space for all actions
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + sum(action_dims), feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList(
            [
                AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim * 2, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)

    def check_action_dims(self, action_dims):
        # currently only support Atari like action space = [A_dim] or Hollow Knight like action space = [2]*A_dim
        if len(action_dims) == 1:
            return True
        for value in action_dims:
            if value != action_dims[0]:
                return False
        return True

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = F.one_hot(action.long(), self.action_choices).float()  # [B, L, A_dim, A_choices]
        action = einops.rearrange(action, "B L A_dim A_choices -> B L (A_dim A_choices)")
        feats = self.stem(torch.cat([samples, action], dim=-1))

        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device="cuda"))

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1] + 1, samples.device)

        action = F.one_hot(action.long(), self.action_choices).float()  # [B, L, A_dim, A_choices]
        action = einops.rearrange(action, "B L A_dim A_choices -> B L (A_dim A_choices)")
        feats = self.stem(torch.cat([samples, action], dim=-1))

        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats
