#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh -- prepare a rented GPU server for retriever training.
#
#   export HF_TOKEN=hf_xxx                      # write access, for pushes
#   export HF_DATASET=username/my-retrieval-dataset
#   export HF_MODEL_REPO=username/legal-e5-base # where checkpoints go
#   bash setup.sh
#
# Keeps the server's preinstalled PyTorch: pip is CONSTRAINED to the installed
# torch version, so installing sentence-transformers cannot silently swap in a
# different build (and a different CUDA) underneath you.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN (needs write access to push checkpoints)}"
: "${HF_DATASET:?export HF_DATASET, e.g. username/my-retrieval-dataset}"
BASE_MODEL="${BASE_MODEL:-intfloat/multilingual-e5-base}"
WORKDIR="${WORKDIR:-$HOME/retriever}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"

mkdir -p "$WORKDIR" "$HF_HOME"
cd "$WORKDIR"

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "== existing torch =="
python - <<'EOF'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"devices {torch.cuda.device_count()} | bf16 {torch.cuda.is_bf16_supported()}")
assert torch.cuda.is_available(), "CUDA not available -- check the image"
EOF

echo "== pinning torch so pip cannot replace it =="
pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > constraints.txt || true
cat constraints.txt

echo "== installing packages =="
pip install --upgrade pip
pip install -c constraints.txt --upgrade \
    "sentence-transformers>=3.3" \
    "transformers>=4.46" \
    "datasets>=3.0" \
    "accelerate>=1.0" \
    "peft>=0.13" \
    tensorboard \
    "huggingface_hub[hf_transfer]"

# confirm torch was not touched
python -c "import torch, sys; print('torch still', torch.__version__)"

echo "== hub login + prefetch (fast transfer) =="
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<EOF
import os
from huggingface_hub import login, whoami
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("logged in as", whoami()["name"])

# Prefetch into HF_HOME so training starts immediately and every GPU process
# reads the same local Arrow cache instead of each downloading.
corpus = load_dataset("${HF_DATASET}", "corpus", split="train")
train  = load_dataset("${HF_DATASET}", "train",  split="train")
print(f"corpus rows: {corpus.num_rows:,}   columns: {corpus.column_names}")
print(f"train  rows: {train.num_rows:,}   columns: {train.column_names}")

SentenceTransformer("${BASE_MODEL}")
print("model cached: ${BASE_MODEL}")
EOF

echo "== accelerate config (all local GPUs, bf16) =="
accelerate config default --mixed_precision bf16
accelerate env | head -20

cat > env.sh <<EOF
export HF_HOME="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASET="$HF_DATASET"
export HF_MODEL_REPO="${HF_MODEL_REPO:-}"
EOF

echo
echo "Done. Before training:  source $WORKDIR/env.sh"
echo "Then:  accelerate launch train_retriever_e5.py --hub-model-id \$HF_MODEL_REPO"
echo "TensorBoard:  tensorboard --logdir runs/ --bind_all   (or the Hub 'Training metrics' tab)"