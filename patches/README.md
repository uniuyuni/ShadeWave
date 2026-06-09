# patches/

サードパーティ・リポジトリへローカル適用するパッチ。

## sam3-macos.patch

公式 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) を macOS
(MPS/CPU) で動かすための差分。`setup.sh` が公式リポジトリを **下記の pin コミット**
で clone した直後に `git apply` する。

- **pin commit**: `8e451d5eb43c817b64ae7577fb7b9ae223db88a9`
  (official/main, SAM 3.1 リリース後の lint 整理まで含む)
- 対象: `sam3/` 配下のソース + `pyproject.toml` のみ（24 ファイル）。
  BPE 語彙 (`sam3/assets/bpe_simple_vocab_16e6.txt.gz`) は公式パッケージに同梱
  されているため、パッチには含めない。

### 主な内容
- `edt.py`: Triton (CUDA 専用) → OpenCV フォールバック
- `model_misc.py`: `get_default_device()` / `tensor_to_device()`（MPS/CPU 選択）
- 各 predictor / encoder / decoder: テンソルのデバイス整合を MPS/CPU 安全に
- `pyproject.toml`: `decord` → `decord2`（Apple Silicon に prebuilt wheel あり）

### パッチの更新方法
公式が進んで pin を上げたいとき:

```bash
# 1. 作業コピーで公式 main をチェックアウトしてパッチを当て、衝突を解消
# 2. 解消後のツリーから差分を取り直す
git diff <new-official-commit>..HEAD -- sam3/ pyproject.toml > patches/sam3-macos.patch
# 3. setup.sh と patches/README.md の pin コミットを更新
```
