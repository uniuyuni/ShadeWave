#!/usr/bin/env bash

set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
  echo "エラー: pixi が見つかりません。https://pixi.sh/latest/#installation を参照してインストールしてください。" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# HTTPS クローン可能な git を選ぶ。pixi 環境の conda git は git-remote-https を
# 欠くことがあり HTTPS clone が "remote helper 'https' aborted session" で失敗する
# ため、git-remote-https を持つ git（多くは system git）を優先的に使う。
select_git() {
  local g
  for g in git /usr/bin/git "$ROOT_DIR/.pixi/envs/default/bin/git"; do
    command -v "$g" >/dev/null 2>&1 || continue
    if ls "$("$g" --exec-path 2>/dev/null)" 2>/dev/null | grep -q '^git-remote-https$'; then
      echo "$g"
      return 0
    fi
  done
  echo "エラー: HTTPS クローン可能な git が見つかりません (git-remote-https が必要)。" >&2
  exit 1
}
GIT="$(select_git)"
echo "git: $GIT ($("$GIT" --version))"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [ ! -d "$target_dir" ]; then
    "$GIT" clone --depth 1 "$repo_url" "$target_dir"
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

# 公式 facebookresearch/sam3 を pin コミットで clone し、macOS(MPS/CPU) 用
# パッチ(patches/sam3-macos.patch)を適用する。詳細は patches/README.md 参照。
ensure_sam3() {
  local pin="8e451d5eb43c817b64ae7577fb7b9ae223db88a9"
  local patch="$ROOT_DIR/patches/sam3-macos.patch"

  if [ ! -d "SAM3" ]; then
    "$GIT" clone https://github.com/facebookresearch/sam3.git SAM3
    "$GIT" -C SAM3 checkout --quiet "$pin"
    # 既に適用済みでない場合のみ当てる（再実行の冪等性）
    if ! "$GIT" -C SAM3 apply --reverse --check "$patch" >/dev/null 2>&1; then
      "$GIT" -C SAM3 apply "$patch"
      echo "SAM3: macOS パッチを適用しました ($patch)"
    fi
  fi
}

mkdir -p checkpoints depth_pro/checkpoints

pixi install

ensure_libraw_enhanced

ensure_sam3

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

if [ ! -f "checkpoints/sam3.1_multiplex.pt" ]; then
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    cat >&2 <<'EOF'
sam3.1_multiplex.pt を Hugging Face から取得するにはアクセストークンが必要です。

  1. トークンをまだお持ちでない場合は、次のページで作成できます（Read で可）。
     https://huggingface.co/settings/tokens

  2. ゲート付きモデルの場合は、利用規約に同意してアクセスを有効にしてください。
     https://huggingface.co/facebook/sam3.1

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
  if ! pixi run python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="facebook/sam3.1", filename="sam3.1_multiplex.pt", local_dir="checkpoints")'; then
    echo '警告: sam3.1_multiplex.pt を取得できませんでした。モデルページでアクセス権を確認し、トークンを再確認してください。' >&2
  fi
fi

if [ ! -f "depth_pro/checkpoints/depth_pro.pt" ]; then
  pixi run curl -fSL -C - 'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt' -o 'depth_pro/checkpoints/depth_pro.pt'
fi

if [ ! -f "checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
  pixi run curl -fSL -C - 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true' -o 'checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf'
fi

if [ ! -f "checkpoints/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" ]; then
  pixi run curl -fSL -C - 'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf?download=true' -o 'checkpoints/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
fi
