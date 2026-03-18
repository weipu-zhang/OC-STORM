#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Boxing Pong Breakout RoadRunner
env_prefix="Boxing"
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --save_frames false \
    --eval_episodes 10 \
    --run_name "${env_prefix}-DEV" \
    --env_name "${env_prefix}NoFrameskip-v4" \
    --seed 42 \
    --config_name "atari_vector_visual"


######################################################

# HornetProtector MageLord MantisLords HKPrime MegaMossCharger Mawlek GodTamer
# GrimmBoss BattleSisters

# eval
# boss_name="HornetProtector"
# CUDA_VISIBLE_DEVICES=0 python -u eval.py \
#     --save_frames false \
#     --eval_episodes 10 \
#     --run_name "${boss_name}-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"
