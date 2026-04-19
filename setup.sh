#!/usr/bin/env bash

set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
  echo "エラー: pixi が見つかりません。https://pixi.sh/latest/#installation を参照してインストールしてください。" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [ ! -d "$target_dir" ]; then
    pixi run git clone --depth 1 "$repo_url" "$target_dir"
  fi
}

ensure_libraw_enhanced() {
  clone_if_missing https://github.com/uniuyuni/libraw_enhanced.git libraw_enhanced

  local marker="libraw_enhanced/external/LibRaw-master/src/metadata/identify.cpp"
  if [ ! -f "$marker" ]; then
    mkdir -p libraw_enhanced/external
    local ver="0.22.1"
    local t
    t="$(mktemp)"
    pixi run curl -fSL "https://github.com/LibRaw/LibRaw/archive/refs/tags/${ver}.tar.gz" -o "$t"
    tar -xzf "$t" -C libraw_enhanced/external
    rm -f "$t"
    rm -rf libraw_enhanced/external/LibRaw-master
    mv "libraw_enhanced/external/LibRaw-${ver}" libraw_enhanced/external/LibRaw-master
  fi

  if [ ! -e "metal" ]; then
    ln -sfn "libraw_enhanced/core/metal" "metal"
  fi
}

mkdir -p checkpoints depth_pro/checkpoints

pixi install

ensure_libraw_enhanced

clone_if_missing https://github.com/uniuyuni/SAM3.git SAM3

# libraw_enhanced は pixi 内でビルドした LibRaw（third_party/libraw-install）にリンクする（システム LibRaw 不要）
echo "LibRaw を third_party/libraw-install にビルドしています..."
pixi run build-libraw

pixi run python -m pip install --upgrade pip "setuptools>=70,<82" wheel
# libraw_enhanced / SAM3 は editable のため別途 install（llvm-openmp / ローカル LibRaw 向けに libraw は --no-build-isolation）
REQ_NO_LOCAL="$(mktemp)"
grep -v '^[[:space:]]*-e[[:space:]].*libraw_enhanced' requirements.txt | \
  grep -v '^[[:space:]]*-e[[:space:]].*SAM3' >"$REQ_NO_LOCAL"
pixi run python -m pip install -r "$REQ_NO_LOCAL"
rm -f "$REQ_NO_LOCAL"
pixi run python -m pip install -e ./libraw_enhanced --no-build-isolation
pixi run python -m pip install -e ./SAM3

clone_if_missing https://github.com/cszn/SCUNet.git SCUNet
clone_if_missing https://github.com/gfacciol/demosaicnet_torch.git demosaicnet_torch

if [ ! -d "icc" ] || [ ! -d "dcp" ] || [ ! -d "luts" ]; then
  pixi run gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1dWrL7ciw5DWlk9zFEBf63Gz9uKsWjJ_W?usp=sharing"
fi

mkdir -p checkpoints/SCUNet
if [ ! -f "checkpoints/SCUNet/scunet_color_real_psnr.pth" ]; then
  pixi run python SCUNet/main_download_pretrained_models.py --models "SCUNet" --model_dir "checkpoints/SCUNet"
fi

if [ ! -f "checkpoints/sam3.pt" ]; then
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    cat >&2 <<'EOF'
sam3.pt を Hugging Face から取得するにはアクセストークンが必要です。

  1. トークンをまだお持ちでない場合は、次のページで作成できます（Read で可）。
     https://huggingface.co/settings/tokens

  2. ゲート付きモデルの場合は、利用規約に同意してアクセスを有効にしてください。
     https://huggingface.co/facebook/sam3

続けてトークンを入力すると、このシェルに HF_TOKEN が設定され、ダウンロードが実行されます（入力は画面に表示されません）。
EOF
    if [ ! -t 0 ]; then
      echo "エラー: 対話入力できません。事前に export HF_TOKEN=... してから実行してください。" >&2
      exit 1
    fi
    HF_TOKEN_INPUT=""
    while [ -z "${HF_TOKEN_INPUT}" ]; do
      read -r -s -p "アクセストークンを入力してください: " HF_TOKEN_INPUT
      echo "" >&2
      if [ -z "${HF_TOKEN_INPUT}" ]; then
        echo "トークンが空です。もう一度入力してください。" >&2
      fi
    done
    export HF_TOKEN="${HF_TOKEN_INPUT}"
    unset HF_TOKEN_INPUT
  fi
  if ! pixi run python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir="checkpoints")'; then
    echo '警告: sam3.pt を取得できませんでした。モデルページでアクセス権を確認し、トークンを再確認してください。' >&2
  fi
fi

if [ ! -f "depth_pro/checkpoints/depth_pro.pt" ]; then
  pixi run curl -fSL -C - 'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt' -o 'depth_pro/checkpoints/depth_pro.pt'
fi
