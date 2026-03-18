import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Categorical
import einops
import einops.layers
import einops.layers.torch
from torch.cuda.amp import autocast

from .modules.functions_losses import SymLogTwoHotLoss
from .modules.attention_blocks import (
    get_causal_mask_with_batch_length,
    get_causal_mask,
    AttentionBlockKVCache,
    PositionwiseFeedForward,
)
from .modules.transformer_model import StochasticTransformerKVCache
from .policy import ActorCritic


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels, out_channels=stem_channels, kernel_size=4, stride=2, padding=1, bias=False
            )
        )
        feature_width = 64 // 2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.SiLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels, out_channels=channels * 2, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.SiLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = einops.rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = einops.rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels * final_feature_width * final_feature_width, bias=False))
        backbone.append(
            einops.layers.torch.Rearrange("B L (C H W) -> (B L) C H W", C=last_channels, H=final_feature_width)
        )
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.SiLU(inplace=True))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels, out_channels=channels // 2, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.SiLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels, out_channels=original_in_channels, kernel_size=4, stride=2, padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = einops.rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class StateDistHead(nn.Module):
    """
    Dist: abbreviation of distribution
    """

    def __init__(self, image_feat_dim, transformer_hidden_dim, latent_width) -> None:
        super().__init__()
        self.latent_width = latent_width
        self.post_head = nn.Sequential(
            PositionwiseFeedForward(image_feat_dim, image_feat_dim * 2, dropout=0.1),
            nn.Linear(image_feat_dim, latent_width * latent_width),
        )
        self.prior_head = nn.Sequential(
            PositionwiseFeedForward(transformer_hidden_dim, transformer_hidden_dim * 2, dropout=0.1),
            nn.Linear(transformer_hidden_dim, latent_width * latent_width),
        )

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.latent_width + (1 - mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = einops.rearrange(logits, "B L Obj (K C) -> B L Obj K C", K=self.latent_width)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = einops.rearrange(logits, "B L Obj (K C) -> B L Obj K C", K=self.latent_width)
        logits = self.unimix(logits)
        return logits


class VisualDistHead(nn.Module):
    """
    Dist: abbreviation of distribution
    """

    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim * stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim * stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1 - mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = einops.rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = einops.rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, feat_dim, num_heads) -> None:
        super().__init__()
        self.reward_token = nn.Parameter(torch.randn(feat_dim))
        self.attention_block = AttentionBlockKVCache(
            feat_dim=feat_dim, hidden_dim=feat_dim * 2, num_heads=num_heads, dropout=0.1
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, feat):
        batch_size, batch_length = feat.shape[:2]
        reward_token = einops.repeat(self.reward_token, "D -> B L 1 D", B=batch_size, L=batch_length)
        feat = torch.cat([reward_token, feat], dim=-2)  # -> B L Obj+1 D
        feat = einops.rearrange(feat, "B L R_Obj D -> (B L) R_Obj D")
        feat, _ = self.attention_block(feat, feat, feat)
        feat = einops.rearrange(feat, "(B L) R_Obj D -> B L R_Obj D", B=batch_size)
        reward_token = feat[:, :, 0]
        reward = self.head(reward_token)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self, feat_dim, num_heads) -> None:
        super().__init__()
        self.termination_token = nn.Parameter(torch.randn(feat_dim))
        self.attention_block = AttentionBlockKVCache(
            feat_dim=feat_dim, hidden_dim=feat_dim * 2, num_heads=num_heads, dropout=0.1
        )
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, feat):
        batch_size, batch_length = feat.shape[:2]
        termination_token = einops.repeat(self.termination_token, "D -> B L 1 D", B=batch_size, L=batch_length)
        feat = torch.cat([termination_token, feat], dim=-2)  # -> B L Obj+1 D
        feat = einops.rearrange(feat, "B L T_Obj D -> (B L) T_Obj D")
        feat, _ = self.attention_block(feat, feat, feat)
        feat = einops.rearrange(feat, "(B L) T_Obj D -> B L T_Obj D", B=batch_size)
        termination_token = feat[:, :, 0]
        termination = self.head(termination_token)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class VisualMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs) ** 2
        loss = einops.reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class VectorMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, state_hat, state):
        loss = (state_hat - state) ** 2
        loss = einops.reduce(loss, "B L Obj D -> B L Obj", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = einops.reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)
        return kl_div, real_kl_div


class OC_CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = Categorical(logits=p_logits)
        q_dist = Categorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(
            p_dist, q_dist
        )  # which gives identical results for Categorical and OneHotCategorical
        kl_div = einops.reduce(kl_div, "B L Obj D -> B L Obj", "sum")  # [Modified] From sum over Obj D to sum over Obj
        real_kl_div = kl_div.mean()

        kl_div = einops.reduce(kl_div, "B L Obj -> Obj", "mean")
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)  # distinguish different objects
        kl_div = kl_div.mean()

        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(
        self,
        action_dims,
        num_objects,
        latent_width,
        transformer_max_length,
        transformer_hidden_dim,
        transformer_num_layers,
        transformer_num_heads,
    ):
        super().__init__()
        self.use_amp = True
        self.amp_tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.action_dims = action_dims
        self.num_objects = num_objects
        self.transformer_hidden_dim = transformer_hidden_dim
        self.latent_width = latent_width
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1

        self.state_encoder = Encoder(in_dim=2048, out_dim=256)
        self.state_decoder = Decoder(in_dim=256, out_dim=2048)

        self.visual_encoder = EncoderBN(in_channels=3, stem_channels=32, final_feature_width=4)
        self.visual_decoder = DecoderBN(
            stoch_dim=1024,
            last_channels=self.visual_encoder.last_channels,
            original_in_channels=3,
            stem_channels=32,
            final_feature_width=4,
        )

        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.latent_width * self.latent_width,
            action_dims=action_dims,
            num_objects=num_objects + 1,  # +1 for visual token
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
        )

        self.state_dist_head = StateDistHead(
            image_feat_dim=256, transformer_hidden_dim=transformer_hidden_dim, latent_width=self.latent_width
        )
        self.visual_dist_head = VisualDistHead(
            image_feat_dim=self.visual_encoder.last_channels * 4 * 4,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=32,
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            feat_dim=transformer_hidden_dim,
            num_heads=transformer_num_heads,
        )
        self.termination_decoder = TerminationDecoder(
            feat_dim=transformer_hidden_dim,
            num_heads=transformer_num_heads,
        )

        self.vector_mse_loss_func = VectorMSELoss()
        self.visual_mse_loss_func = VisualMSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.state_categorical_kl_div_loss = OC_CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.visual_categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode_obs(self, state, obs):
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            # state VAE
            state_embedding = self.state_encoder(state)
            state_post_logits = self.state_dist_head.forward_post(state_embedding)
            state_sample = self.straight_through_gradient(state_post_logits, sample_mode="random_sample")
            state_flattened_sample = einops.rearrange(state_sample, "B L Obj K C -> B L Obj (K C)")

            # visual VAE
            obs_embedding = self.visual_encoder(obs)
            obs_post_logits = self.visual_dist_head.forward_post(obs_embedding)
            obs_sample = self.straight_through_gradient(obs_post_logits, sample_mode="random_sample")
            obs_flattened_sample = einops.rearrange(obs_sample, "B L K C -> B L (K C)")

        return state_flattened_sample, obs_flattened_sample

    def calc_last_hidden(self, state_latent, obs_latent, action):
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            temporal_mask = get_causal_mask(state_latent)
            dist_feat, state_dist_feat, obs_dist_feat = self.storm_transformer(
                state_latent, obs_latent, action, temporal_mask
            )
            last_hidden = dist_feat[:, -1:]
        return last_hidden

    def predict_next(self, last_state_flattened_sample, last_obs_flattened_sample, action, log_video=True):
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            dist_feat, state_dist_feat, obs_dist_feat = self.storm_transformer.forward_with_kv_cache(
                last_state_flattened_sample, last_obs_flattened_sample, action
            )
            state_prior_logits = self.state_dist_head.forward_prior(state_dist_feat)
            obs_prior_logits = self.visual_dist_head.forward_prior(obs_dist_feat)

            state_prior_sample = self.straight_through_gradient(state_prior_logits, sample_mode="random_sample")
            state_flattened_prior_sample = einops.rearrange(state_prior_sample, "B L Obj K C -> B L Obj (K C)")
            obs_prior_sample = self.straight_through_gradient(obs_prior_logits, sample_mode="random_sample")
            obs_flattened_prior_sample = einops.rearrange(obs_prior_sample, "B L K C -> B L (K C)")

            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return None, reward_hat, termination_hat, state_flattened_prior_sample, obs_flattened_prior_sample, dist_feat

    def straight_through_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        """
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        """
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            state_latent_size = (imagine_batch_size, imagine_batch_length + 1, self.num_objects, 16 * 16)
            obs_latent_size = (imagine_batch_size, imagine_batch_length + 1, 32 * 32)
            hidden_size = (
                imagine_batch_size,
                imagine_batch_length + 1,
                self.num_objects + 1,
                self.transformer_hidden_dim,
            )  # +1 for visual token
            action_size = (imagine_batch_size, imagine_batch_length, len(self.action_dims))
            scalar_size = (imagine_batch_size, imagine_batch_length)

            self.state_latent_buffer = torch.zeros(state_latent_size, dtype=dtype, device="cuda")
            self.obs_latent_buffer = torch.zeros(obs_latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=torch.int32, device="cuda")
            self.log_prob_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=torch.int32, device="cuda")

    def imagine_data(
        self,
        agent: ActorCritic,
        sample_state,
        sample_obs,
        sample_action,
        imagine_batch_size,
        imagine_batch_length,
        log_video,
        logger,
    ):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.amp_tensor_dtype)
        obs_hat_list = []

        context_length = sample_state.shape[1]
        self.storm_transformer.reset_kv_cache_list(
            imagine_batch_size, context_length + imagine_batch_length, dtype=self.amp_tensor_dtype
        )
        # context
        context_state_latent, context_obs_latent = self.encode_obs(sample_state, sample_obs)
        for i in range(context_length):
            last_obs_hat, last_reward_hat, last_termination_hat, last_state_latent, last_obs_latent, last_dist_feat = (
                self.predict_next(
                    context_state_latent[:, i : i + 1],
                    context_obs_latent[:, i : i + 1],
                    sample_action[:, i : i + 1],
                    log_video=log_video,
                )
            )
        self.state_latent_buffer[:, 0:1] = last_state_latent
        self.obs_latent_buffer[:, 0:1] = last_obs_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action, log_prob = agent.sample_with_log_prob(
                torch.cat(
                    [
                        einops.rearrange(self.state_latent_buffer[:, i : i + 1], "B L Obj D -> B L (Obj D)"),
                        self.obs_latent_buffer[:, i : i + 1],
                        einops.rearrange(self.hidden_buffer[:, i : i + 1], "B L Obj D -> B L (Obj D)"),
                    ],
                    dim=-1,
                )
            )
            self.action_buffer[:, i : i + 1] = action
            self.log_prob_buffer[:, i : i + 1] = log_prob

            last_obs_hat, last_reward_hat, last_termination_hat, last_state_latent, last_obs_latent, last_dist_feat = (
                self.predict_next(
                    self.state_latent_buffer[:, i : i + 1],
                    self.obs_latent_buffer[:, i : i + 1],
                    self.action_buffer[:, i : i + 1],
                    log_video=log_video,
                )
            )

            self.state_latent_buffer[:, i + 1 : i + 2] = last_state_latent
            self.obs_latent_buffer[:, i + 1 : i + 2] = last_obs_latent
            self.hidden_buffer[:, i + 1 : i + 2] = last_dist_feat
            self.reward_hat_buffer[:, i : i + 1] = last_reward_hat
            self.termination_hat_buffer[:, i : i + 1] = last_termination_hat

        latent_hidden = torch.cat(
            [
                einops.rearrange(self.state_latent_buffer, "B L Obj D -> B L (Obj D)"),
                self.obs_latent_buffer,
                einops.rearrange(self.hidden_buffer, "B L Obj D -> B L (Obj D)"),
            ],
            dim=-1,
        )
        return (
            latent_hidden,
            self.action_buffer,
            self.log_prob_buffer,
            self.reward_hat_buffer,
            self.termination_hat_buffer,
        )

    def update(self, state, obs, action, reward, termination, logger=None):
        self.train()
        batch_size, batch_length = state.shape[:2]

        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            # state VAE
            state_embedding = self.state_encoder(state)
            state_post_logits = self.state_dist_head.forward_post(state_embedding)
            state_sample = self.straight_through_gradient(state_post_logits, sample_mode="random_sample")
            state_flattened_sample = einops.rearrange(state_sample, "B L Obj K C -> B L Obj (K C)")
            state_hat = self.state_decoder(state_flattened_sample)

            # visual VAE
            obs_embedding = self.visual_encoder(obs)
            obs_post_logits = self.visual_dist_head.forward_post(obs_embedding)
            obs_sample = self.straight_through_gradient(obs_post_logits, sample_mode="random_sample")
            obs_flattened_sample = einops.rearrange(obs_sample, "B L K C -> B L (K C)")
            obs_hat = self.visual_decoder(obs_flattened_sample)

            # spatial-temporal transformer
            temporal_mask = get_causal_mask_with_batch_length(batch_length, state_flattened_sample.device)
            dist_feat, state_dist_feat, obs_dist_feat = self.storm_transformer(
                state_flattened_sample, obs_flattened_sample, action, temporal_mask
            )

            # prior logits
            state_prior_logits = self.state_dist_head.forward_prior(state_dist_feat)
            obs_prior_logits = self.visual_dist_head.forward_prior(obs_dist_feat)
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            state_reconstruction_loss = self.vector_mse_loss_func(state_hat, state)
            obs_reconstruction_loss = self.visual_mse_loss_func(obs_hat, obs)
            reconstruction_loss = state_reconstruction_loss + obs_reconstruction_loss
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination.float())
            # dyn-rep loss
            state_dynamics_loss, state_dynamics_real_kl_div = self.state_categorical_kl_div_loss(
                state_post_logits[:, 1:].detach(), state_prior_logits[:, :-1]
            )
            state_representation_loss, state_representation_real_kl_div = self.state_categorical_kl_div_loss(
                state_post_logits[:, 1:], state_prior_logits[:, :-1].detach()
            )
            obs_dynamics_loss, obs_dynamics_real_kl_div = self.visual_categorical_kl_div_loss(
                obs_post_logits[:, 1:].detach(), obs_prior_logits[:, :-1]
            )
            obs_representation_loss, obs_representation_real_kl_div = self.visual_categorical_kl_div_loss(
                obs_post_logits[:, 1:], obs_prior_logits[:, :-1].detach()
            )
            dynamics_loss = state_dynamics_loss + obs_dynamics_loss
            representation_loss = state_representation_loss + obs_representation_loss
            dynamics_real_kl_div = state_dynamics_real_kl_div + obs_dynamics_real_kl_div
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss
            )

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
                with torch.no_grad():
                    # reward MSE loss, distinguish zero & non-zero situation
                    reward_hat_value = self.symlog_twohot_loss_func.decode(reward_hat)  # [B, L]
                    reward_hat_value, reward = reward_hat_value.flatten(), reward.flatten()
                    zero_reward_idices = reward == 0
                    if sum(zero_reward_idices) > 0:
                        reward_zero_mse_loss = (reward_hat_value[zero_reward_idices] - reward[zero_reward_idices]) ** 2
                        reward_zero_mse_loss = reward_zero_mse_loss.mean()
                        logger.log("WorldModel/reward_zero_mse_loss", reward_zero_mse_loss.item())
                    else:
                        logger.log("WorldModel/reward_zero_mse_loss", None)
                    if sum(zero_reward_idices) < len(reward):
                        reward_nonzero_mse_loss = (
                            reward_hat_value[~zero_reward_idices] - reward[~zero_reward_idices]
                        ) ** 2
                        reward_nonzero_mse_loss = reward_nonzero_mse_loss.mean()
                        logger.log("WorldModel/reward_nonzero_mse_loss", reward_nonzero_mse_loss.item())
                    else:
                        logger.log("WorldModel/reward_nonzero_mse_loss", None)

                    # termination F1 score
                    termination_hat_value = termination_hat > 0  # [B, L]
                    termination_hat_value, termination = termination_hat_value.flatten(), termination.flatten().bool()
                    tp = (termination_hat_value & termination).sum().float()
                    fp = (termination_hat_value & ~termination).sum().float()
                    tn = (~termination_hat_value & ~termination).sum().float()
                    fn = (~termination_hat_value & termination).sum().float()
                    if tp + fn > 0:
                        positive_recall = tp / (tp + fn)
                        logger.log("WorldModel/termination_positive_recall", positive_recall.item())
                    else:
                        logger.log("WorldModel/termination_positive_recall", None)
                    if fp + tn > 0:
                        negative_recall = tn / (fp + tn)
                        logger.log("WorldModel/termination_negative_recall", negative_recall.item())
                    else:
                        logger.log("WorldModel/termination_negative_recall", None)

            logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
            logger.log("WorldModel/state_reconstruction_loss", state_reconstruction_loss.item())
            logger.log("WorldModel/obs_reconstruction_loss", obs_reconstruction_loss.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            logger.log("WorldModel/state_dynamic_loss", state_dynamics_loss.item())
            logger.log("WorldModel/state_dynamic_real_kl_div", state_dynamics_real_kl_div.item())
            logger.log("WorldModel/obs_dynamic_loss", obs_dynamics_loss.item())
            logger.log("WorldModel/obs_dynamic_real_kl_div", obs_dynamics_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
