#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [ ! -d "$target_dir" ]; then
    git clone --depth 1 "$repo_url" "$target_dir"
  fi
}

fetch_libraw_metal() {
  if [ -d "metal" ]; then
    return
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"

  git clone --depth 1 --filter=blob:none --sparse https://github.com/uniuyuni/libraw_enhanced.git "$tmp_dir/libraw_enhanced"
  git -C "$tmp_dir/libraw_enhanced" sparse-checkout set core/metal
  mv "$tmp_dir/libraw_enhanced/core/metal" "$ROOT_DIR/metal"
  rm -rf "$tmp_dir"
}

mkdir -p checkpoints

pixi install
pixi run python -m pip install --upgrade pip setuptools wheel
pixi run python -m pip install -r requirements.txt

clone_if_missing https://github.com/uniuyuni/SAM3.git SAM3
clone_if_missing https://github.com/cszn/SCUNet.git SCUNet
fetch_libraw_metal

if [ ! -d "icc" ] || [ ! -d "dcp" ] || [ ! -d "luts" ]; then
  pixi run gdown --folder "https://drive.google.com/drive/folders/1dWrL7ciw5DWlk9zFEBf63Gz9uKsWjJ_W?usp=sharing"
fi

if [ ! -d "checkpoints/_SCUNet" ]; then
  pixi run python SCUNet/main_download_pretrained_models.py --models "SCUNet" --model_dir "checkpoints/_SCUNet"
fi

ln -sfn _SCUNet checkpoints/SCUNet

if [ ! -f "checkpoints/sam3.pt" ]; then
  pixi run curl -L -C - "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt" -o "checkpoints/sam3.pt"
fi
