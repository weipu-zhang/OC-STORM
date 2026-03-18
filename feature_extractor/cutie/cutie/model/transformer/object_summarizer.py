from typing import List, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor.cutie.cutie.model.transformer.positional_encoding import PositionalEncoding
import einops


# @torch.jit.script
def _weighted_pooling(masks: torch.Tensor, value: torch.Tensor, logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # value: B*num_objects*H*W*value_dim
    # logits: B*num_objects*H*W*num_summaries
    # masks: B*num_objects*H*W*num_summaries: 1 if allowed
    weights = logits.sigmoid() * masks  # W, eq8
    # B*num_objects*num_summaries*value_dim
    sums = torch.einsum("bkhwq,bkhwc->bkqc", weights, value)  # eq6 numerator
    # B*num_objects*H*W*num_summaries -> B*num_objects*num_summaries*1
    area = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)

    # visualize weights >>>
    # import matplotlib.pyplot as plt
    # print(weights.shape, sums.shape, area.shape)
    # fig, axes = plt.subplots(4, 4)
    # for i in range(4):
    #     for j in range(4):
    #         axes[i, j].imshow(weights[0, 0, :, :, i*4+j].cpu().numpy())
    # plt.show()
    # exit()
    # <<< visualize weights

    # [MODIFY] [Deprecated] This is not called every frames, so not quite suitable
    # flattened_weights = einops.rearrange(weights, 'b k h w n -> b k (h w) n')

    # B*num_objects*num_summaries*value_dim
    return sums, area


class ObjectSummarizer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(
                self.embed_dim, scale=self.pixel_pe_scale, temperature=self.pixel_pe_temperature
            )

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

        # [MODIFY] [Deprecated] This is not called every frames, so not quite suitable
        # self.flattened_weights_cache = None
        # self.counter = 0

    def forward(
        self, masks: torch.Tensor, value: torch.Tensor, need_weights: bool = False
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode="area")
        masks = masks.unsqueeze(-1)
        inv_masks = 1 - masks
        repeated_masks = torch.cat(
            [
                masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
                inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
            ],
            dim=-1,
        )

        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe

        with torch.cuda.amp.autocast(enabled=False):
            value = value.float()
            feature = self.feature_pred(value)  # U, eq7
            logits = self.weights_pred(value)  # W
            sums, area = _weighted_pooling(repeated_masks, feature, logits)  # bkqc, bkq1
            # [MODIFY]
            # self.flattened_weights_cache = flattened_weights
            # print("summ", self.counter)
            # self.counter += 1

        summaries = torch.cat([sums, area], dim=-1)  # Why cat? Eq6 says is weighted avg pool
        if need_weights:
            return summaries, logits
        else:
            return summaries, None
