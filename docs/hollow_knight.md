# Hollow Knight Setup

> 中文版本：[hollow_knight_zh.md](hollow_knight_zh.md)

> **Game version notice:** Please use Hollow Knight version **1.5.78.11833**. The latest version of the game lacks mod API support and compatibility with most mods. We will update this project if the modding ecosystem migrates to the new version at scale; otherwise, we will keep things as they are.

## Hardware requirements

A GPU equal to or stronger than a 3090 is required; otherwise, achieving the target execution frequency of 9 FPS will be difficult.

## General description

Hollow Knight training uses `ray` for communication between the game node and the training node. The game runs on one node, while model training and inference run on another. In most experiments in the paper, both are hosted on a single Windows machine: a Hyper-V virtual machine runs the game, WSL runs PyTorch training and inference, and `ray` passes observations and actions between them.

This distributed setup has several practical advantages:
1. Training in WSL is faster than running the same PyTorch workload in PowerShell.
2. On Windows, sending keyboard inputs to the game window requires that window to stay in the foreground. Without a virtual machine or a dedicated machine for the game, training would block normal desktop use and make monitoring much less convenient.
3. The setup is naturally scalable: as shown in `train_async.py`, a more powerful non-Windows node can be used for faster training.

## Environment setup

We developed a mod to extract hit and damage signals for reward computation. To install the mod, first install the mod API from https://github.com/hk-modding/api/releases. There are more detailed and intuitive tutorials for this step in the Hollow Knight modding community, so we do not repeat them here. After that, install the provided `HKRLEnv` mod from `game_mod/HKRLEnv/`.
The source code of the `HKRLEnv` mod is at [here](https://github.com/weipu-zhang/HKRLEnv).

The game environment and the training environment must use the same version of `python` and `ray`.

1. Install the game-environment dependencies on the game node (e.g., the virtual machine) with `pip install -r requirements-game.txt`.
2. Launch `./scripts/start_ray.sh` on the training node.
3. Launch `./scripts/win_start_ray.ps1` on the virtual machine, and update the conda environment name and IP address for your setup.
4. Optional: run `ray status` to verify the connection.
5. Launch training with `./scripts/train.sh`. Both `train.py` and `train_async.py` support Hollow Knight. `train.py` skips training during episodes for higher inference frequency and performs policy improvement afterward; `train_async.py` requires two GPUs — one continuously runs policy improvement while the other handles inference.
