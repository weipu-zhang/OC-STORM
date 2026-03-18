import argparse
import cv2
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import colorama
import random
import shutil
import os
import time
import warnings

from utils import tools
import importlib


def train_ratio_scheduling(episode_length, step, total_steps, min_train_ratio, max_train_ratio):
    """
    Train ratio scheduling
    """
    middle_step = step - episode_length / 2
    current_progress = middle_step / total_steps
    current_train_ratio = min_train_ratio + (max_train_ratio - min_train_ratio) * current_progress
    return current_train_ratio


if __name__ == "__main__":
    # ignore warnings
    warnings.filterwarnings("ignore")
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
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_name", type=str, required=True)  # please pass "None" if not using dynamic importing
    args = parser.parse_args()
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    tools.seed_np_torch(seed=args.seed)
    # create log/ckpt folder
    logger = tools.Logger(run_name=args.run_name)  # tensorboard writer
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

    # train >>>
    # reset envs and variables
    episode_count = 1
    episode_return = 0
    current_obs, current_info = env.reset()
    context_state = deque(maxlen=params.eval_context_length + 1)  # +1 for the current_state
    context_action = deque(maxlen=params.eval_context_length)

    # sample and train
    for current_sample_step in tqdm(range(params.max_sample_steps)):
        current_state, visualization_obs = feature_extractor.extract_features(current_obs)

        # policy part >>>
        if replay_buffer.ready():
            context_state.append(current_state)  # first append the current state
            if len(context_action) == 0:  # First step of the episode, no context
                action = np.zeros(action_space.dim, dtype=np.int32)
            else:
                if random.random() > 0.01:
                    action = agent.sample_policy(context_state, context_action, greedy=False)
                else:
                    action = action_space.sample()
            context_action.append(action)  # finally append the action
        else:  # warmup
            action = action_space.sample()
        # <<< policy part

        obs, reward, terminated, truncated, info = env.step(action)
        replay_buffer.append(current_state, action, reward, terminated or info["life_loss"], episode_count)

        # visualization >>>
        if opencv_visualization:
            cv2.imshow("visualization_obs", cv2.cvtColor(visualization_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        # <<< visualization

        # update current_obs, current_info and episode_return
        episode_return += reward
        current_obs = obs
        current_info = info

        # save model for evaluation
        if current_sample_step % params.save_every_steps == 0 and current_sample_step > 0:
            print(
                colorama.Fore.GREEN
                + f"Saving log model at total steps {current_sample_step}"
                + colorama.Style.RESET_ALL
            )
            torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/agent_{current_sample_step}.pth")

        # useful option when debugging
        # if terminated or truncated or current_sample_step > 80:
        #     replay_buffer.warmup_length = 0
        if terminated or truncated:
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

            logger.log("sample/episode_return", episode_return)
            logger.log("sample/episode_length", episode_length)
            if "HollowKnight" in args.env_name:
                logger.log("sample/win_battle", current_info["win_battle"])
                logger.log("sample/health_remains", current_info["health"])

            logger.log("replay_buffer/length", len(replay_buffer))
            # <<< logs

            # reset episode_return for next episode
            episode_return = 0

            if replay_buffer.ready():
                print(colorama.Fore.CYAN + "\nEpisode done, start training" + colorama.Style.RESET_ALL)
                current_train_ratio = train_ratio_scheduling(
                    episode_length,
                    current_sample_step,
                    params.max_sample_steps,
                    params.min_train_ratio,
                    params.max_train_ratio,
                )
                logger.log("train/train_ratio", current_train_ratio)
                training_steps = int(current_train_ratio * episode_length)
                training_start_time = time.time()
                for _ in tqdm(range(training_steps)):
                    agent.update(
                        replay_buffer,
                        training_steps,
                        params.batch_size,
                        params.batch_length,
                        params.imagine_batch_size,
                        params.imagine_context_length,
                        params.imagine_batch_length,
                        logger,
                    )
                training_time = time.time() - training_start_time
                if (
                    training_time < 6 and "HollowKnight" in args.env_name
                ):  # Hollow Knight, see the else branch, level 3 boss's episode length can be too short
                    if training_time < 6 + 8 and "HKPrime" in args.env_name:
                        # TODO: in HKPrime, there would be an extra 8sec post-swing if the boss is defeated
                        # here we sleep for 14 seconds despite win or lose, should be optimized later
                        time.sleep(6 + 8 - training_time)
                    else:
                        time.sleep(6 - training_time)
            else:
                print(colorama.Fore.BLUE + "\nBuffer not warmed up, skip training" + colorama.Style.RESET_ALL)
                if "HollowKnight" in args.env_name:
                    time.sleep(6)  # wait for game to load, for Hollow Knight

            # reset envs and variables
            print(colorama.Fore.BLUE + "\nTraining done, reset env" + colorama.Style.RESET_ALL)
            current_obs, current_info = env.reset()
            # save latest model every episode
            print(colorama.Fore.GREEN + f"Saving latest model at step {current_sample_step}" + colorama.Style.RESET_ALL)
            torch.save(agent.state_dict(), f"runs/{args.run_name}/ckpt/latest_agent.pth")

    # <<< train
    if "HollowKnight" in args.env_name:
        env.release_action()
        print(colorama.Fore.RED + "Execution done, release keyboard" + colorama.Style.RESET_ALL)
