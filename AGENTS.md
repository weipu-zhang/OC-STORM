# Quick Orientation for Code Agents

## Start Here

- `scripts/train.sh`: the most useful launcher example in this repository. It shows the common command patterns for training and evaluation.
- `train.py`: main synchronous training entrypoint.
- `train_async.py`: async training entrypoint used by the main Hollow Knight workflow.
- `eval.py`: evaluation entrypoint.

The entry scripts all take the same core routing inputs:

- `--run_name`
- `--env_name`
- `--seed`
- `--config_name`

Training and evaluation runs write logs and checkpoints under `runs/<run_name>/`. The selected config file is also copied there as `runs/<run_name>/config.py` to make runs easier to reproduce.

## Configuration System

The project uses dynamic config loading. The entry scripts import:

- `configs.<config_name>`

For example:

- `--config_name atari_vector_visual` loads `configs/atari_vector_visual.py`

Each config file is the main assembly point for an experiment. A config is expected to provide:

- `Params`: hyperparameters and feature flags
- `build(env_name, seed)`: constructs the training components

In practice, `build(...)` wires together the environment, action space, feature extractor, replay buffer, and agent. If you want to understand what one run is actually using, start from the chosen config file rather than the entry script.

## How to Trace a Run

The shortest path for understanding a run is:

`scripts/train.sh` or CLI args -> `train.py` / `train_async.py` / `eval.py` -> `configs/<config_name>.py` -> imports inside that config

Those imports usually lead you to the real implementation in:

- `agents/`
- `envs/`
- `feature_extractor/`

This repository currently includes setups for Atari and Hollow Knight, but the navigation pattern is the same across environments.

## Repository Map

- `configs/`: experiment assembly and configuration entrypoints. Start here when you need to see which implementations are combined for a run.
- `agents/`: learning logic, world models, policies, and update code. Most algorithm changes belong here.
- `envs/`: environment builders and wrappers. Change this when you need to modify observation processing, action spaces, or environment integration.
- `feature_extractor/`: visual encoders and feature extraction pipelines. Change this when working on image-to-latent processing or extractor loading.
- `utils/`: shared utilities such as logging, seeding, and replay-buffer helpers.
- `scripts/`: runnable shell entrypoints and local workflow helpers.
- `runs/`: generated outputs such as logs, copied configs, and checkpoints. Treat this as run output, not source code.
- `segmentation_masks/`: supporting assets used by some visual pipelines.

## Editing Guidance

1. Find the target experiment entrypoint in `scripts/train.sh` or in the command you were given.
2. Open the selected file in `configs/` and inspect its imports and `build(...)` function.
3. Change the referenced implementation module in `agents/`, `envs/`, or `feature_extractor/` instead of hardcoding behavior into `train.py`, `train_async.py`, or `eval.py`.
4. When adding a new experiment, prefer adding a new config file in `configs/` and selecting it with `--config_name`.
5. Keep entry scripts thin. Use config files for wiring and module selection.
