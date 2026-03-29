#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Boxing Pong etc.
env_prefix="Boxing"
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --run_name "${env_prefix}-DEV" \
    --env_name "${env_prefix}NoFrameskip-v4" \
    --seed 42 \
    --config_name "atari_vector_visual"

######################################################

# HornetProtector MageLord MantisLords HKPrime MegaMossCharger Mawlek GodTamer
# GrimmBoss BattleSisters

# boss_name="HornetProtector"
# CUDA_VISIBLE_DEVICES=0 python -u train.py \
#     --run_name "${boss_name}-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"

# boss_name="MantisLords"
# python -u train_async.py \
#     --run_name "${boss_name}-async-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"

