import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import einops
import copy
import numpy as np
from torch.cuda.amp import autocast

from .modules.functions_losses import SymLogTwoHotLoss


class EMAScalar:
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage * len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    gamma_return = torch.zeros((batch_size, batch_length + 1), dtype=dtype, device="cuda")
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = (
            rewards[:, t]
            + gamma * inv_termination[:, t] * (1 - lam) * values[:, t]
            + gamma * inv_termination[:, t] * lam * gamma_return[:, t + 1]
        )
    return gamma_return[:, :-1]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super().__init__()
        mlp = [nn.Linear(input_dim, hidden_dim, bias=False), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for layer in range(num_layers - 1):
            mlp.extend([nn.Linear(hidden_dim, hidden_dim, bias=False), nn.LayerNorm(hidden_dim), nn.SiLU()])
        self.mlp = nn.Sequential(*mlp, nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dims, gamma, lambd, entropy_coef) -> None:
        super().__init__()
        assert self.check_action_dims(action_dims), (
            "Currently only support Atari like action space = [A_dim] or Hollow Knight like action space = [2]*A_dim"
        )
        self.action_dims = action_dims
        self.action_choices = action_dims[0]

        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.clip_coef = 0.2
        self.use_amp = True
        self.amp_tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        # self.amp_tensor_dtype = torch.float16 if self.use_amp else torch.float32

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        self.actor = MLP(feat_dim, hidden_dim, sum(action_dims), num_layers)  # multi-discrete action space
        self.critic = MLP(feat_dim, hidden_dim, 255, num_layers)  # symlog twohot representation
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def check_action_dims(self, action_dims):
        # currently only support Atari like action space = [A_dim] or Hollow Knight like action space = [2]*A_dim
        if len(action_dims) == 1:
            return True
        for value in action_dims:
            if value != action_dims[0]:
                return False
        return True

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        logits = einops.rearrange(logits, "B L (A_dim A_choices) -> B L A_dim A_choices", A_choices=self.action_choices)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        logits = einops.rearrange(logits, "B L (A_dim A_choices) -> B L A_dim A_choices", A_choices=self.action_choices)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()

        latent = einops.rearrange(latent, "B L Obj D -> B L (Obj D)")
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            logits = self.policy(latent)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action

    @torch.no_grad()
    def sample_with_log_prob(self, latent, temperature=1.0):
        self.eval()

        latent = einops.rearrange(latent, "B L Obj D -> B L (Obj D)")
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            logits = self.policy(latent)
            dist = distributions.Categorical(logits=logits / temperature)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = einops.reduce(log_prob, "B L A_dim -> B L", "sum")
        return action, log_prob

    def sample_as_env_action(self, latent, greedy=False) -> np.ndarray:
        action = self.sample(latent, greedy)
        return action.detach().cpu().numpy()

    def update(self, latent, action, old_log_prob, reward, termination, logger=None):
        """
        Update policy and value model
        """
        self.train()

        latent = einops.rearrange(latent, "B L Obj D -> B L (Obj D)")
        with torch.autocast(device_type="cuda", dtype=self.amp_tensor_dtype, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            log_prob = einops.reduce(log_prob, "B L A_dim -> B L", "sum")
            entropy = dist.entropy()
            entropy = einops.reduce(entropy, "B L A_dim -> B L", "sum")

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound - lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return - value[:, :-1]) / norm_ratio

            # off-policy loss
            log_ratio = log_prob - old_log_prob
            ratio = torch.exp(log_ratio)
            policy_loss = -ratio * norm_advantage.detach()
            policy_loss = policy_loss.mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log("ActorCritic/policy_loss", policy_loss.item())
            logger.log("ActorCritic/value_loss", value_loss.item())
            logger.log("ActorCritic/entropy_loss", entropy_loss.item())
            logger.log("ActorCritic/S", S.item())
            logger.log("ActorCritic/norm_ratio", norm_ratio.item())
            logger.log("ActorCritic/total_loss", loss.item())
