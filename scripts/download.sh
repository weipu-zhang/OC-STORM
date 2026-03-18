#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

weights_dir="${root_dir}/feature_extractor/cutie/weights"
segmentation_archive="${root_dir}/segmentation_masks.tar"

mkdir -p "${weights_dir}"

# Download CUTIE model weights used by the visual feature extractor.
wget -O "${weights_dir}/coco_lvis_h18_itermask.pth" \
  "https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth"
wget -O "${weights_dir}/cutie-small-mega.pth" \
  "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth"

# Download segmentation masks to the project root and optionally extract them in place.
wget -O "${segmentation_archive}" \
  "https://github.com/weipu-zhang/OC-STORM/releases/download/v1.0/segmentation_masks.tar"

extract_choice="Y"
if [ -t 0 ]; then
  read -r -p "Extract segmentation_masks.tar to ${root_dir}? [Y/n] " extract_choice
fi

extract_choice="${extract_choice:-Y}"
if [[ "${extract_choice}" =~ ^[Yy]$ ]]; then
  tar -xf "${segmentation_archive}" -C "${root_dir}"
  echo "Extracted segmentation masks under ${root_dir}"
else
  echo "Skipped extraction for ${segmentation_archive}"
fi

echo "Downloaded weights to ${weights_dir}"
echo "Downloaded segmentation archive to ${segmentation_archive}"
