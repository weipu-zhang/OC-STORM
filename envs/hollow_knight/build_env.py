import ray
from typing import Tuple
import shutil
import tempfile

from envs.hollow_knight.env_wrapper import HKEnv, IndependentActionSpace, LocalAbstractHKEnv


def build_hollow_knight_env(boss_name, obs_size, target_fps) -> Tuple[LocalAbstractHKEnv, IndependentActionSpace]:
    temp_dir = tempfile.mkdtemp()
    shutil.copytree(
        ".",
        temp_dir,
        ignore=shutil.ignore_patterns(
            "atari-labeling-tools",
            "human_play",
            "runs",
            "archive_runs",
            "feature_extractor",
            "segmentation_masks",
            "eval_videos",
            "tmp",
            "*.log",
            "*.ipynb",
            "*.sh",
            "*.h5",
            ".git",
        ),
        dirs_exist_ok=True,
    )
    ray.init(runtime_env={"working_dir": temp_dir}, _metrics_export_port=None)

    possible_actions = ["w", "a", "s", "d", "j", "k", "l", "i"]
    env = HKEnv.remote(boss_name=boss_name, obs_size=obs_size, target_fps=target_fps, possible_actions=possible_actions)
    local_env_wrapper = LocalAbstractHKEnv(env)
    action_space = IndependentActionSpace(possible_actions)
    return local_env_wrapper, action_space


def build_hollow_knight_env_ban_spell(
    boss_name, obs_size, target_fps
) -> Tuple[LocalAbstractHKEnv, IndependentActionSpace]:
    temp_dir = tempfile.mkdtemp()
    shutil.copytree(
        ".",
        temp_dir,
        ignore=shutil.ignore_patterns(
            "atari-labeling-tools",
            "human_play",
            "runs",
            "archive_runs",
            "feature_extractor",
            "segmentation_masks",
            "eval_videos",
            "tmp",
            "*.log",
            "*.ipynb",
            "*.sh",
            "*.h5",
            ".git",
        ),
        dirs_exist_ok=True,
    )
    ray.init(runtime_env={"working_dir": temp_dir}, _metrics_export_port=None)

    possible_actions = ["w", "a", "s", "d", "j", "k", "l"]  # no i for spells
    env = HKEnv.remote(boss_name=boss_name, obs_size=obs_size, target_fps=target_fps, possible_actions=possible_actions)
    local_env_wrapper = LocalAbstractHKEnv(env)
    action_space = IndependentActionSpace(possible_actions)
    return local_env_wrapper, action_space
