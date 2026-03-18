#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tensorboard --logdir "${PROJECT_ROOT}/runs" --port 6006 --host 0.0.0.0
