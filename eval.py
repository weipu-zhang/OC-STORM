import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os
import time
from PIL import Image
import importlib

from utils import tools


def train_ratio_scheduling(episode_length, step, total_steps, min_train_ratio, max_train_ratio):
    """
    Train ratio scheduling
    """
    middle_step = step - episode_length / 2
    current_progress = middle_step / total_steps
    current_train_ratio = min_train_ratio + (max_train_ratio - min_train_ratio) * current_progress
    return current_train_ratio


def str2bool(v):  # for argparse, store_true is not flexible in multi-line scripts
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    # ignore warnings
    # import warnings
    # warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # detect if use visualiation, remote servers don't have graphical interface
    if "HKRL_LOCAL_DEVICE" in os.environ:
        opencv_visualization = True
    else:
        opencv_visualization = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--save_frames", type=str2bool, required=True)
    parser.add_argument("--eval_episodes", type=int, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    tools.seed_np_torch(seed=args.seed)
    # create log/ckpt folder
    logger = tools.Logger(run_name=args.run_name + "-eval")  # tensorboard writer
    os.makedirs(f"runs/{args.run_name}/ckpt", exist_ok=True)  # ckpt dir

    # load config
    if args.config_name != "None":
        # pass
        module_name = f"configs.{args.config_name}"
        config_module = importlib.import_module(module_name)
        build = getattr(config_module, "build")
        Params = getattr(config_module, "Params")
        print(colorama.Fore.RED + f"Using {args.config_name}.py" + colorama.Style.RESET_ALL)
        shutil.copy(f"configs/{args.config_name}.py", f"runs/{args.run_name}/config.py")
    else:
        # print(colorama.Fore.YELLOW + "[WARN] config_path not used, the importing is following the code" + colorama.Style.RESET_ALL)
        ###############################################################################################
        # mannual import, IDE friendly
        from configs.hollow_knight_vector_visual import build, Params

        print(colorama.Fore.RED + "Using hollow_knight_vector_visual.py" + colorama.Style.RESET_ALL)
        shutil.copy("configs/hollow_knight_vector_visual.py", f"runs/{args.run_name}/config.py")
        ###############################################################################################

    # build all components
    params, env, action_space, feature_extractor, replay_buffer, agent = build(env_name=args.env_name, seed=args.seed)

    # load agent model
    agent_ckpt_path = f"runs/{args.run_name}/ckpt/latest_agent.pth"
    agent.load_state_dict(torch.load(agent_ckpt_path))
    print(colorama.Fore.GREEN + f"Loaded agent model from {agent_ckpt_path}" + colorama.Style.RESET_ALL)
    agent.eval()

    # train >>>
    # reset envs and variables
    episode_count = 1
    save_folder_prefix = f"eval_videos/{args.run_name}"
    os.makedirs(f"{save_folder_prefix}/episode_{episode_count}/", exist_ok=True)
    episode_frame_list = []
    episode_return_list, win_battle_list, health_remains_list = [], [], []
    episode_return = 0
    current_obs, current_info = env.reset()
    context_state = deque(maxlen=params.eval_context_length + 1)  # +1 for the current_state
    context_action = deque(maxlen=params.eval_context_length)

    print(colorama.Fore.YELLOW + f"\nTotal episodes: {args.eval_episodes}" + colorama.Style.RESET_ALL)
    # sample and train
    while True:
        current_state, visualization_obs = feature_extractor.extract_features(current_obs)

        # policy part >>>
        context_state.append(current_state)  # first append the current state
        if len(context_action) == 0:  # First step of the episode, no context
            action = np.zeros(action_space.dim, dtype=np.int32)
        else:
            action = agent.sample_policy(context_state, context_action, greedy=True)
        context_action.append(action)  # finally append the action
        # <<< policy part

        obs, reward, terminated, truncated, info = env.step(action)

        episode_frame_list.append(visualization_obs)

        # visualization >>>
        if opencv_visualization:
            cv2.imshow("visualization_obs", cv2.cvtColor(visualization_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        # <<< visualization

        # update current_obs, current_info and episode_return
        episode_return += reward
        current_obs = obs
        current_info = info

        # useful option when debugging
        # if terminated or current_sample_step > 80:
        #     replay_buffer.warmup_length = 0
        if terminated:
            # add the last frame
            current_state, visualization_obs = feature_extractor.extract_features(current_obs)
            episode_frame_list.append(visualization_obs)

            # clear feature extractor memory
            feature_extractor.reset()
            # clear context for next episode
            context_state.clear()
            context_action.clear()

            # logs >>>
            episode_length = current_info["episode_frame_number"] // params.frame_skip
            print(colorama.Fore.YELLOW + f"\nEpisode {episode_count} done" + colorama.Style.RESET_ALL)
            print("Return: " + colorama.Fore.YELLOW + f"{episode_return}" + colorama.Style.RESET_ALL)
            episode_count += 1

            episode_return_list.append(episode_return)
            logger.log("sample/episode_return", episode_return)
            logger.log("sample/episode_length", episode_length)
            if "HollowKnight" in args.env_name:
                win_battle_list.append(current_info["win_battle"])
                logger.log("sample/win_battle", current_info["win_battle"])
                health_remains_list.append(current_info["health"])
                logger.log("sample/health_remains", current_info["health"])
            # <<< logs

            # reset episode_return for next episode
            episode_return = 0

            # save video
            if args.save_frames == True:
                print(colorama.Fore.BLUE + "Saving frames" + colorama.Style.RESET_ALL)
                for idx in tqdm(range(len(episode_frame_list))):
                    frame = episode_frame_list[idx]
                    cv2.imwrite(
                        f"{save_folder_prefix}/episode_{episode_count - 1}/{idx}.png",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    )
                print(colorama.Fore.BLUE + "frames saved" + colorama.Style.RESET_ALL)
            episode_frame_list = []  # clear frame list

            # break if enough episodes
            if episode_count > args.eval_episodes:
                break

            if "HollowKnight" in args.env_name:
                time.sleep(10)  # wait for game to load, for Hollow Knight

            # reset envs and variables
            print(colorama.Fore.BLUE + "\nReset env" + colorama.Style.RESET_ALL)
            os.makedirs(f"{save_folder_prefix}/episode_{episode_count}/", exist_ok=True)
            current_obs, current_info = env.reset()
    # <<< train

    # save episode_return list as csv
    os.makedirs(f"eval_results", exist_ok=True)
    if "HollowKnight" in args.env_name:
        with open(f"eval_results/{args.run_name}_episode_return.csv", "w") as f:
            f.write("episode, return, win_battle, health_remains\n")
            for idx in range(len(episode_return_list)):
                f.write(f"{idx}, {episode_return_list[idx]}, {win_battle_list[idx]}, {health_remains_list[idx]}\n")
    else:
        with open(f"eval_results/{args.run_name}_episode_return.csv", "w") as f:
            f.write("episode, return\n")
            for idx in range(len(episode_return_list)):
                f.write(f"{idx}, {episode_return_list[idx]}\n")
