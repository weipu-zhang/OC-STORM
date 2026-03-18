import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .attention_blocks import get_vector_mask
from .attention_blocks import PositionalEncoding1D, PositionalEncoding2D, AttentionBlock, AttentionBlockKVCache


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dims, num_objects, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        assert self.check_action_dims(action_dims), (
            "Currently only support Atari like action space = [A_dim] or Hollow Knight like action space = [2]*A_dim"
        )
        self.action_dims = action_dims
        self.action_choices = action_dims[0]  # only support same action space for all actions
        self.num_objects = num_objects
        self.feat_dim = feat_dim
        self.num_layers = num_layers

        # mix state latent and action
        self.state_stem = nn.Sequential(
            nn.Linear(256 + sum(action_dims), feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        # mix visual latent and action
        self.visual_stem = nn.Sequential(
            nn.Linear(1024 + sum(action_dims), feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )

        # self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.position_encoding = PositionalEncoding2D(
            max_length=max_length, num_objects=self.num_objects, embed_dim=feat_dim
        )
        self.layer_stack = nn.ModuleList(
            [
                # *2 due to spatial and temporal attention
                AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim * 2, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers * 2)
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

    def forward(self, state_samples, obs_samples, action, mask):
        """
        Normal forward pass
        state_samples: [B, L, Obj, 16*16]
        obs_samples: [B, L, 32*32]
        action: [B, L, A_dim] int
        """
        batch_size = state_samples.shape[0]

        action = F.one_hot(action.long(), self.action_choices).float()  # [B, L, A_dim, A_choices]
        action = einops.rearrange(action, "B L A_dim A_choices -> B L (A_dim A_choices)")
        # obs
        obs_feats = self.visual_stem(torch.cat([obs_samples, action], dim=-1))
        obs_feats = einops.rearrange(obs_feats, "B L D -> B L 1 D")
        # state
        action = einops.repeat(action, "B L A -> B L Obj A", Obj=state_samples.shape[2])
        state_feats = self.state_stem(torch.cat([state_samples, action], dim=-1))
        # concat
        feats = torch.cat([state_feats, obs_feats], dim=2)

        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer_idx in range(0, len(self.layer_stack), 2):
            spatial_block = self.layer_stack[layer_idx]
            temporal_block = self.layer_stack[layer_idx + 1]

            # The attention block will treat the second dimension as the token dimension
            feats = einops.rearrange(feats, "B L Obj D -> (B L) Obj D")
            feats, attn = spatial_block(feats, feats, feats)
            feats = einops.rearrange(feats, "(B L) Obj D -> B L Obj D", B=batch_size)

            feats = einops.rearrange(feats, "B L Obj D -> (B Obj) L D")
            feats, attn = temporal_block(feats, feats, feats, mask)
            feats = einops.rearrange(feats, "(B Obj) L D -> B L Obj D", B=batch_size)

        # split state and obs
        state_feats, obs_feats = feats[:, :, :-1], feats[:, :, -1]
        return feats, state_feats, obs_feats

    def reset_kv_cache_list(self, batch_size, expected_batch_length, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        self.kv_cache_step_memory_list = []
        # only cache temporal attention
        for idx in range(self.num_layers):
            self.kv_cache_list.append(
                torch.empty(
                    size=(batch_size * self.num_objects, expected_batch_length, self.feat_dim),
                    dtype=dtype,
                    device="cuda",
                )
            )
            self.kv_cache_step_memory_list.append(0)

    def forward_with_kv_cache(self, state_samples, obs_samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        assert state_samples.shape[1] == 1
        batch_size = state_samples.shape[0]
        mask = get_vector_mask(self.kv_cache_step_memory_list[0] + 1, state_samples.device)

        action = F.one_hot(action.long(), self.action_choices).float()  # [B, L, A_dim, A_choices]
        action = einops.rearrange(action, "B L A_dim A_choices -> B L (A_dim A_choices)")
        # obs
        obs_feats = self.visual_stem(torch.cat([obs_samples, action], dim=-1))
        obs_feats = einops.rearrange(obs_feats, "B L D -> B L 1 D")
        # state
        action = einops.repeat(action, "B L A -> B L Obj A", Obj=state_samples.shape[2])
        state_feats = self.state_stem(torch.cat([state_samples, action], dim=-1))
        # concat
        feats = torch.cat([state_feats, obs_feats], dim=2)

        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_step_memory_list[0])
        feats = self.layer_norm(feats)

        for layer_idx in range(0, len(self.layer_stack), 2):
            macro_layer_idx = layer_idx // 2
            spatial_block = self.layer_stack[layer_idx]
            temporal_block = self.layer_stack[layer_idx + 1]

            # The attention block will treat the second dimension as the token dimension
            feats = einops.rearrange(feats, "B L Obj D -> (B L) Obj D")
            feats, attn = spatial_block(feats, feats, feats)
            feats = einops.rearrange(feats, "(B L) Obj D -> B L Obj D", B=batch_size)

            feats = einops.rearrange(feats, "B L Obj D -> (B Obj) L D")
            self.kv_cache_list[macro_layer_idx][:, self.kv_cache_step_memory_list[macro_layer_idx], :] = (
                einops.rearrange(feats, "B_Obj 1 D -> B_Obj D")
            )
            self.kv_cache_step_memory_list[macro_layer_idx] += 1
            feats, attn = temporal_block(
                feats,
                self.kv_cache_list[macro_layer_idx][:, : self.kv_cache_step_memory_list[macro_layer_idx], :],
                self.kv_cache_list[macro_layer_idx][:, : self.kv_cache_step_memory_list[macro_layer_idx], :],
                mask,
            )
            feats = einops.rearrange(feats, "(B Obj) L D -> B L Obj D", B=batch_size)

        # split state and obs
        state_feats, obs_feats = feats[:, :, :-1], feats[:, :, -1]
        return feats, state_feats, obs_feats
