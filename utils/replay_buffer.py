import numpy as np
import random
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
from concurrent.futures import ThreadPoolExecutor


class ReplayBuffer:
    def __init__(
        self, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.float32, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        # mix uniform sample
        prob = 0.5 * prob + 0.5 / sample_limit
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination, episode):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    @torch.no_grad()
    def dry_sample(self, batch_size, batch_length):
        """
        For testing only
        """
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return indices

    def dry_append(self, episode):
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length


class VisualReplayBuffer:
    def __init__(
        self, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        # mix uniform sample
        prob = 0.5 * prob + 0.5 / sample_limit
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        # convert uint8 obs to float32
        obs = obs.to(torch.float32) / 255

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination, episode):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        # convert float32 obs to uint8
        obs = obs * 255
        obs = obs.to(torch.uint8)

        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    @torch.no_grad()
    def dry_sample(self, batch_size, batch_length):
        """
        For testing only
        """
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return indices

    def dry_append(self, episode):
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length


class ReplayBufferVectorPlusVisual:
    """
    Cutie object vector + visual observation
    """

    def __init__(
        self, state_shape, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.state_buffer = torch.empty((max_length, *state_shape), dtype=torch.float32, device="cuda")
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        prob = 0.5 * prob + 0.5 / sample_limit  # mix uniform
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        state = torch.stack([self.state_buffer[idx : idx + batch_length] for idx in indices])
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        # convert uint8 obs to float32
        obs = obs.to(torch.float32) / 255

        return state, obs, action, reward, termination

    def append(self, state_obs, action, reward, termination, episode):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        state, obs = state_obs  # tuple
        # convert float32 obs to uint8
        obs = obs * 255
        obs = obs.to(torch.uint8)

        self.state_buffer[self.last_pointer] = state
        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length
