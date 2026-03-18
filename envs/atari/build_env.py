import ale_py

import gymnasium
from gymnasium.core import Env
import envs.atari.env_wrapper as env_wrapper
import numpy as np
from typing import Tuple


class ActionSpace:
    """
    Single dimension action space
    """

    def __init__(self, choices_per_dim, expand_dim) -> None:
        self.dim = 1
        self.choices_per_dim = choices_per_dim
        self.expand_dim = expand_dim

    def sample(self):
        action = np.random.randint(self.choices_per_dim)
        if self.expand_dim:
            action = np.array([action])
        return action


def build_single_atari_env(env_name, seed, image_size=None) -> Tuple[Env, ActionSpace]:
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.StochasticSeedEnvWrapper(env)
    env = env_wrapper.NoopResetWrapper(env, noop_max=30)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    if image_size is not None:
        env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfoWrapper(env)
    env = env_wrapper.SqueezeActionDimWrapper(env)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=5000)  # truncate episode

    action_space = ActionSpace(env.action_space.n, expand_dim=True)
    return env, action_space


def build_single_atari_env_no_squeeze(env_name, seed, image_size=None) -> Tuple[Env, ActionSpace]:
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.StochasticSeedEnvWrapper(env)
    env = env_wrapper.NoopResetWrapper(env, noop_max=30)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    if image_size is not None:
        env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfoWrapper(env)

    action_space = ActionSpace(env.action_space.n, expand_dim=True)
    return env, action_space


def build_single_atari_env_no_max_last2(env_name, seed, image_size=None) -> Env:
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=4)
    env = env_wrapper.StochasticSeedEnvWrapper(env)
    if image_size is not None:
        env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env
