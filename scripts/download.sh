#!/usr/bin/env bash

set -euo pipefail

weights_dir="feature_extractor/cutie/weights"
mkdir -p "${weights_dir}"

wget -O "${weights_dir}/coco_lvis_h18_itermask.pth" \
  "https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth"
wget -O "${weights_dir}/cutie-small-mega.pth" \
  "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth"

echo "Downloaded weights to ${weights_dir}"
