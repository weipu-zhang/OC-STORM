import argparse
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
import sys
import importlib
from unittest.mock import MagicMock
import torch.multiprocessing as mp

from utils import tools


def train_ratio_scheduling(episode_length, step, total_steps, min_train_ratio, max_train_ratio):
    """
    Train ratio scheduling
    """
    middle_step = step - episode_length / 2
    current_progress = middle_step / total_steps
    current_train_ratio = min_train_ratio + (max_train_ratio - min_train_ratio) * current_progress
    return current_train_ratio


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, tuple):
        return tuple(to_cpu(v) for v in x)
    return x


def learner_worker(
    env_name,
    run_name,
    config_name,
    seed,
    data_queue,
    action_space_proxy,
    buffer_warm_up,
    max_sample_steps,
    num_train_steps_per_episode=1000,
    ckpt_path=None,
):
    # Set device to GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Redirect stdout and stderr to worker.log
    # first mv the log file to worker.log.old
    if os.path.exists("worker.log"):
        os.rename("worker.log", "worker.log.old")
    log_file = open("worker.log", "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    # Mock envs and feature_extractor to avoid initialization in learner process
    sys.modules["envs.hollow_knight.build_env"] = MagicMock()
    sys.modules["envs.hollow_knight.build_env"].build_hollow_knight_env.return_value = (
        None,
        action_space_proxy,
    )
    sys.modules["envs.hollow_knight.build_env"].build_hollow_knight_env_ban_spell.return_value = (
        None,
        action_space_proxy,
    )
    sys.modules["feature_extractor.cutie.build_feature_extractor"] = MagicMock()
    sys.modules["feature_extractor.visual.build_feature_extractor"] = MagicMock()

    # Load config
    if config_name != "None":
        module_name = f"configs.{config_name}"
        config_module = importlib.import_module(module_name)
        build = getattr(config_module, "build")
    else:
        from configs.hollow_knight_vector_visual import build

    # Build (only agent and replay_buffer are used)
    params, _, _, _, replay_buffer, agent = build(env_name=env_name, seed=seed)

    # Load checkpoint if provided
    if ckpt_path is not None:
        agent.load_state_dict(torch.load(ckpt_path))
        print(colorama.Fore.CYAN + f"Learner loaded agent model from {ckpt_path}" + colorama.Style.RESET_ALL)
    else:
        print(
            colorama.Fore.YELLOW
            + "[WARN] Learner: ckpt_path not provided, training will start from scratch"
            + colorama.Style.RESET_ALL
        )

    # Use TensorboardLogger which does not clear the directory
    logger = tools.Logger(run_name=run_name + "-train")

    print(colorama.Fore.GREEN + "Learner process started on GPU 1" + colorama.Style.RESET_ALL)

    os.makedirs(f"runs/{run_name}/ckpt", exist_ok=True)

    training_steps = 0

    # Wait for buffer to be ready before starting training
    print(colorama.Fore.YELLOW + "Waiting for replay buffer to be ready..." + colorama.Style.RESET_ALL)
    while True:
        # Drain queue to fill buffer
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
                if data == "STOP":
                    print("Learner received STOP signal before training started.")
                    log_file.close()
                    return
                replay_buffer.append(*data)
            except:
                break

        # Check if buffer is ready
        if replay_buffer.ready():
            print(colorama.Fore.GREEN + "Replay buffer is ready! Starting training..." + colorama.Style.RESET_ALL)
            break

        # Wait and print status
        print(
            colorama.Fore.CYAN
            + f"Pending for training... {len(replay_buffer)}/{params.buffer_warm_up}"
            + colorama.Style.RESET_ALL
        )
        time.sleep(10)

    # Main training loop
    while True:
        # Train for num_train_steps_per_episode with tqdm progress bar
        for _ in tqdm(range(num_train_steps_per_episode), desc="Training", unit="step", colour="cyan"):
            # Drain queue
            got_data = False
            while not data_queue.empty():
                try:
                    data = data_queue.get_nowait()
                    if data == "STOP":
                        print("Learner received STOP signal.")
                        log_file.close()
                        return
                    replay_buffer.append(*data)
                    got_data = True
                except:
                    break

            # Train
            agent.update(
                replay_buffer,
                1,  # unused
                params.batch_size,
                params.batch_length,
                params.imagine_batch_size,
                params.imagine_context_length,
                params.imagine_batch_length,
                logger,
            )
            training_steps += 1

            # Save checkpoint at regular intervals
            if training_steps % params.save_every_steps == 0:
                step_ckpt_path = f"runs/{run_name}/ckpt/agent_step_{training_steps}.pth"
                temp_path = f"runs/{run_name}/ckpt/temp_agent.pth"
                torch.save(agent.state_dict(), temp_path)
                os.replace(temp_path, step_ckpt_path)
                print(
                    colorama.Fore.GREEN
                    + f"Learner saved checkpoint at step {training_steps}"
                    + colorama.Style.RESET_ALL
                )

        # Save model after each num_train_steps_per_episode (latest)
        latest_path = f"runs/{run_name}/ckpt/latest_agent.pth"
        temp_path = f"runs/{run_name}/ckpt/temp_agent.pth"

        # Atomic save for latest
        torch.save(agent.state_dict(), temp_path)
        os.replace(temp_path, latest_path)
        print(colorama.Fore.GREEN + f"Learner saved latest model at {training_steps} steps" + colorama.Style.RESET_ALL)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

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

    # disable ray metrics collection
    os.environ["RAY_ENABLE_METRICS_COLLECTION"] = "0"

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_name", type=str, required=True)  # please pass "None" if not using dynamic importing
    parser.add_argument("--ckpt_path", type=str, required=False)
    args = parser.parse_args()
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    tools.seed_np_torch(seed=args.seed)
    # create log/ckpt folder
    logger = tools.Logger(run_name=args.run_name + "-sample")  # tensorboard writer
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
        shutil.copy(
            "configs/hollow_knight_vector_visual.py",
            f"runs/{args.run_name}/config.py",
        )
        ###############################################################################################

    # build all components
    # Main process uses GPU 0
    params, env, action_space, feature_extractor, replay_buffer, agent = build(env_name=args.env_name, seed=args.seed)

    if args.ckpt_path is not None:
        agent.load_state_dict(torch.load(args.ckpt_path))
        print(colorama.Fore.CYAN + f"Loaded agent model from {args.ckpt_path}" + colorama.Style.RESET_ALL)
    else:
        print(
            colorama.Fore.YELLOW
            + "[WARN] ckpt_path not provided, the training will start from scratch"
            + colorama.Style.RESET_ALL
        )

    # Start Learner Process
    data_queue = mp.Queue()
    learner = mp.Process(
        target=learner_worker,
        args=(
            args.env_name,
            args.run_name,
            args.config_name,
            args.seed,
            data_queue,
            action_space,
            params.buffer_warm_up,
            params.max_sample_steps,
            params.num_train_steps_per_episode,
            args.ckpt_path,
        ),
    )
    learner.start()

    # train >>>
    # reset envs and variables
    episode_count = 1
    episode_return = 0
    current_obs, current_info = env.reset()
    context_state = deque(maxlen=params.eval_context_length + 1)  # +1 for the current_state
    context_action = deque(maxlen=params.eval_context_length)

    try:
        # sample and train
        for current_sample_step in tqdm(range(params.max_sample_steps)):
            current_state, visualization_obs = feature_extractor.extract_features(current_obs)

            # policy part >>>
            # sample from policy if the agent is loaded or the replay buffer is ready (using warm up param)
            use_policy = current_sample_step > params.buffer_warm_up or (args.ckpt_path is not None)

            if use_policy:
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

            # Send to learner (move tensors to cpu)
            # replay_buffer.append(current_state, action, reward, terminated or info["life_loss"], episode_count)
            data_queue.put(
                (
                    to_cpu(current_state),
                    action,
                    reward,
                    terminated or info["life_loss"],
                    episode_count,
                )
            )

            # visualization >>>
            if opencv_visualization:
                cv2.imshow(
                    "visualization_obs",
                    cv2.cvtColor(visualization_obs, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)
            # <<< visualization

            # update current_obs, current_info and episode_return
            episode_return += reward
            current_obs = obs
            current_info = info

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

                # logger.log("replay_buffer/length", len(replay_buffer)) # Main doesn't know length
                # <<< logs

                # reset episode_return for next episode
                episode_return = 0

                # Reload latest model from learner if available
                latest_ckpt = f"runs/{args.run_name}/ckpt/latest_agent.pth"
                if os.path.exists(latest_ckpt):
                    try:
                        # Load to GPU 0
                        state_dict = torch.load(
                            latest_ckpt,
                            map_location=f"cuda:{torch.cuda.current_device()}",
                        )
                        agent.load_state_dict(state_dict)
                        print(colorama.Fore.MAGENTA + "Reloaded latest agent from learner" + colorama.Style.RESET_ALL)
                    except Exception as e:
                        print(colorama.Fore.RED + f"Failed to reload agent: {e}" + colorama.Style.RESET_ALL)

                # reset envs and variables
                print(colorama.Fore.BLUE + "\nEpisode done, reset env" + colorama.Style.RESET_ALL)
                current_obs, current_info = env.reset()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        data_queue.put("STOP")
        learner.join()

    # <<< train
    if "HollowKnight" in args.env_name:
        env.release_action()
        print(colorama.Fore.RED + "Execution done, release keyboard" + colorama.Style.RESET_ALL)
