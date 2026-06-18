#!/usr/bin/env bash

set -euo pipefail

if ! command -v pixi >/dev/null 2>&1; then
  echo "エラー: pixi が見つかりません。https://pixi.sh/latest/#installation を参照してインストールしてください。" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
EXTERNAL_DIR="$ROOT_DIR/external"
mkdir -p "$EXTERNAL_DIR"

# pixi 環境の conda git は git-remote-https を欠き、HTTPS clone が
#   "remote helper 'https' aborted session"
# で失敗する。pip が内部で git clone https を呼ぶ場合も同じ。
# GIT_EXEC_PATH を system git のヘルパーディレクトリに向けることで
# pixi git / pip 内部 git を含むすべての git 呼び出しを一括で修正する。
_sys_git_exec="$(/usr/bin/git --exec-path 2>/dev/null)"
if [ -n "$_sys_git_exec" ] && [ -f "$_sys_git_exec/git-remote-https" ]; then
  export GIT_EXEC_PATH="$_sys_git_exec"
  echo "GIT_EXEC_PATH → $_sys_git_exec (git-remote-https available)"
else
  echo "警告: /usr/bin/git に git-remote-https が見つかりません。HTTPS clone が失敗する可能性があります。" >&2
fi
unset _sys_git_exec

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [ ! -d "$target_dir" ]; then
    git clone --depth 1 "$repo_url" "$target_dir"
  fi
}

download_file() {
  local url="$1"
  local output="$2"
  shift 2

  SSL_CERT_FILE= CURL_CA_BUNDLE= REQUESTS_CA_BUNDLE= \
    /usr/bin/curl -fSL "$@" "$url" -o "$output"
}

ensure_libraw_enhanced() {
  local repo_dir="$EXTERNAL_DIR/libraw_enhanced"
  clone_if_missing https://github.com/uniuyuni/libraw_enhanced.git "$repo_dir"

  local marker="$repo_dir/external/LibRaw-master/src/metadata/identify.cpp"
  if [ ! -f "$marker" ]; then
    mkdir -p "$repo_dir/external"
    local ver="0.22.1"
    local t
    t="$(mktemp)"
    download_file "https://github.com/LibRaw/LibRaw/archive/refs/tags/${ver}.tar.gz" "$t"
    tar -xzf "$t" -C "$repo_dir/external"
    rm -f "$t"
    rm -rf "$repo_dir/external/LibRaw-master"
    mv "$repo_dir/external/LibRaw-${ver}" "$repo_dir/external/LibRaw-master"
  fi

  if [ ! -e "metal" ]; then
    ln -sfn "external/libraw_enhanced/core/metal" "metal"
  fi
}

# 公式 facebookresearch/sam3 を pin コミットで clone し、macOS(MPS/CPU) 用
# パッチ(patches/sam3-macos.patch)を適用する。詳細は patches/README.md 参照。
ensure_sam3() {
  local pin="8e451d5eb43c817b64ae7577fb7b9ae223db88a9"
  local patch="$ROOT_DIR/patches/sam3-macos.patch"
  local repo_dir="$EXTERNAL_DIR/SAM3"

  if [ ! -d "$repo_dir" ]; then
    git clone https://github.com/facebookresearch/sam3.git "$repo_dir"
    git -C "$repo_dir" checkout --quiet "$pin"
    # 既に適用済みでない場合のみ当てる（再実行の冪等性）
    if ! git -C "$repo_dir" apply --reverse --check "$patch" >/dev/null 2>&1; then
      git -C "$repo_dir" apply "$patch"
      echo "SAM3: macOS パッチを適用しました ($patch)"
    fi
  fi
}

ensure_external_repos() {
  clone_if_missing https://github.com/uniuyuni/radiance_denoise.git "$EXTERNAL_DIR/radiance_denoise"
  clone_if_missing https://github.com/cszn/SCUNet.git "$EXTERNAL_DIR/SCUNet"
  clone_if_missing https://github.com/gfacciol/demosaicnet_torch.git "$EXTERNAL_DIR/demosaicnet_torch"
  clone_if_missing https://github.com/uniuyuni/SCUNet_CoreML.git "$EXTERNAL_DIR/SCUNet_CoreML"
}

ensure_depth_pro() {
  clone_if_missing https://github.com/apple/ml-depth-pro.git "$EXTERNAL_DIR/depth_pro"
}

mkdir -p checkpoints

pixi install

ensure_libraw_enhanced

ensure_sam3

ensure_external_repos

ensure_depth_pro

pixi run python -m pip install --upgrade pip "setuptools>=70,<82" wheel
# requirements.txt を pip 依存の正本にする。libraw_enhanced は Apple clang と
# --no-build-isolation、depth_pro は NumPy 2 系を維持するため --no-deps で
# 入れるので、どちらも専用コマンドに分ける。
REQ_NO_LOCAL="$(mktemp)"
cleanup_requirements_tmp() {
  rm -f "$REQ_NO_LOCAL"
}
trap cleanup_requirements_tmp EXIT
grep -v -E '^[[:space:]]*-e[[:space:]].*external/(libraw_enhanced|depth_pro)' requirements.txt >"$REQ_NO_LOCAL"
pixi run python -m pip install -r "$REQ_NO_LOCAL"
cleanup_requirements_tmp
trap - EXIT
pixi run python -m pip install -e ./external/depth_pro --no-deps

# libraw_enhanced は pixi 内でビルドした LibRaw（third_party/libraw-install）にリンクする（システム LibRaw 不要）
echo "LibRaw を third_party/libraw-install にビルドしています..."
pixi run build-libraw

# pixi の conda clang (arm64-apple-darwin20.0.0-clang++) は新しい macOS SDK (26.x)
# と組み合わせると "could not build module 'Darwin'" および pixi ncurses ヘッダと
# macOS SDK curses.h の "conflicting types for 'unctrl'" でビルドが失敗する。
# system clang (Apple Clang) は SDK を完全に認識するためこれらの問題が起きない。
CC=/usr/bin/clang CXX=/usr/bin/clang++ \
  pixi run python -m pip install -e ./external/libraw_enhanced --no-build-isolation --force-reinstall --no-deps --no-cache-dir

pixi run build-denoise-native
pixi run install-effect-backends

if [ ! -d "icc" ] || [ ! -d "dcp" ] || [ ! -d "luts" ]; then
  pixi run python -m gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1dWrL7ciw5DWlk9zFEBf63Gz9uKsWjJ_W?usp=sharing"
fi

mkdir -p checkpoints/SCUNet
if [ ! -f "checkpoints/SCUNet/scunet_color_real_psnr.pth" ]; then
  pixi run python external/SCUNet/main_download_pretrained_models.py --models "SCUNet" --model_dir "checkpoints/SCUNet"
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

if [ ! -f "external/depth_pro/checkpoints/depth_pro.pt" ]; then
  (
    cd "external/depth_pro"
    SSL_CERT_FILE= CURL_CA_BUNDLE= REQUESTS_CA_BUNDLE= pixi run bash get_pretrained_models.sh
  )
fi

if [ ! -f "checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
  download_file 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true' 'checkpoints/qwen2.5-1.5b-instruct-q4_k_m.gguf' -C -
fi

if [ ! -f "checkpoints/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" ]; then
  download_file 'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf?download=true' 'checkpoints/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf' -C -
fi
