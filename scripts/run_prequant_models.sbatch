#!/bin/bash
#SBATCH --job-name=llm-energy-prequant
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Runs three pre-quantized large models sequentially:
#   1. Qwen2.5-32B-Instruct GPTQ-Int4 (~19GB download)
#   2. Mixtral-8x7B-Instruct GPTQ-Int4 (~24GB download)
#   3. Llama-3.3-70B-Instruct AWQ-Int4 (~40GB download)
# Each model's cache is cleaned after benchmarking to free disk for the next.

set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate camorl

export HF_HOME=$HOME/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export NTFY_TOPIC="llm-energy-framework-127"
mkdir -p slurm_logs

cd /home/daniel.cregg/llm-energy-framework

echo "=== Job $SLURM_JOB_ID: Pre-quantized large models ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "==="

# ---------- Model 1: Qwen2.5-32B GPTQ-Int4 ----------
MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
OUTDIR="results/qwen2.5-32b-gptq"
CACHE_DIR="Qwen--Qwen2.5-32B-Instruct-GPTQ-Int4"

AVAIL_GB=$(df --output=avail -BG $HOME | tail -1 | tr -d ' G')
echo ""
echo "=== Model 1: Qwen2.5-32B GPTQ-Int4 ($(date)) ==="
echo "Available disk: ${AVAIL_GB}GB"

if [ "$AVAIL_GB" -lt 25 ]; then
    echo "SKIP: Only ${AVAIL_GB}GB free, need 25GB for Qwen2.5-32B"
    curl -s -H "Tags: warning" -d "Qwen2.5-32B skipped: only ${AVAIL_GB}GB free" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
else
    curl -s -H "Tags: hourglass" -d "Starting Qwen2.5-32B GPTQ-Int4" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1

    python3 -m src.benchmark \
        --model "$MODEL" \
        --precision gptq \
        --batch-sizes 1 4 8 16 \
        --n-runs 10 \
        --output-dir "$OUTDIR"

    if [ $? -eq 0 ]; then
        curl -s -H "Priority: high" -H "Tags: tada" \
          -d "Qwen2.5-32B GPTQ-Int4 complete!" \
          https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    else
        curl -s -H "Priority: urgent" -H "Tags: warning" \
          -d "Qwen2.5-32B GPTQ-Int4 FAILED" \
          https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    fi

    echo "Cleaning cache: $HF_HOME/hub/models--${CACHE_DIR}"
    rm -rf "$HF_HOME/hub/models--${CACHE_DIR}"
    echo "Disk after cleanup: $(df -h $HOME | tail -1 | awk '{print $4 " free"}')"
fi

# ---------- Model 2: Mixtral-8x7B GPTQ-Int4 ----------
MODEL="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
OUTDIR="results/mixtral-8x7b-gptq"
CACHE_DIR="TheBloke--Mixtral-8x7B-Instruct-v0.1-GPTQ"

AVAIL_GB=$(df --output=avail -BG $HOME | tail -1 | tr -d ' G')
echo ""
echo "=== Model 2: Mixtral-8x7B GPTQ-Int4 ($(date)) ==="
echo "Available disk: ${AVAIL_GB}GB"

if [ "$AVAIL_GB" -lt 30 ]; then
    echo "SKIP: Only ${AVAIL_GB}GB free, need 30GB for Mixtral-8x7B"
    curl -s -H "Tags: warning" -d "Mixtral-8x7B skipped: only ${AVAIL_GB}GB free" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
else
    curl -s -H "Tags: hourglass" -d "Starting Mixtral-8x7B GPTQ-Int4" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1

    python3 -m src.benchmark \
        --model "$MODEL" \
        --precision gptq \
        --batch-sizes 1 4 8 16 \
        --n-runs 10 \
        --output-dir "$OUTDIR"

    if [ $? -eq 0 ]; then
        curl -s -H "Priority: high" -H "Tags: tada" \
          -d "Mixtral-8x7B GPTQ-Int4 complete!" \
          https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    else
        curl -s -H "Priority: urgent" -H "Tags: warning" \
          -d "Mixtral-8x7B GPTQ-Int4 FAILED" \
          https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    fi

    echo "Cleaning cache: $HF_HOME/hub/models--${CACHE_DIR}"
    rm -rf "$HF_HOME/hub/models--${CACHE_DIR}"
    echo "Disk after cleanup: $(df -h $HOME | tail -1 | awk '{print $4 " free"}')"
fi

# ---------- Model 3: Llama-3.3-70B AWQ-Int4 ----------
MODEL="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
OUTDIR="results/llama-3.3-70b-awq"
CACHE_DIR="ibnzterrell--Meta-Llama-3.3-70B-Instruct-AWQ-INT4"

AVAIL_GB=$(df --output=avail -BG $HOME | tail -1 | tr -d ' G')
echo ""
echo "=== Model 3: Llama-3.3-70B AWQ-Int4 ($(date)) ==="
echo "Available disk: ${AVAIL_GB}GB"

if [ "$AVAIL_GB" -lt 42 ]; then
    echo "SKIP: Only ${AVAIL_GB}GB free, need 42GB for Llama-3.3-70B"
    curl -s -H "Tags: warning" -d "Llama-3.3-70B skipped: only ${AVAIL_GB}GB free" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
else
    curl -s -H "Tags: hourglass" -d "Starting Llama-3.3-70B AWQ-Int4" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1

    python3 -m src.benchmark \
        --model "$MODEL" \
        --precision awq \
        --batch-sizes 1 4 8 16 \
        --n-runs 10 \
        --output-dir "$OUTDIR"

    if [ $? -eq 0 ]; then
        curl -s -H "Priority: high" -H "Tags: tada" \
          -d "Llama-3.3-70B AWQ-Int4 complete!" \
          https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    else
        curl -s -H "Priority: urgent" -H "Tags: warning" \
          -d "Llama-3.3-70B AWQ-Int4 FAILED" \
          https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    fi

    echo "Cleaning cache: $HF_HOME/hub/models--${CACHE_DIR}"
    rm -rf "$HF_HOME/hub/models--${CACHE_DIR}"
    echo "Disk after cleanup: $(df -h $HOME | tail -1 | awk '{print $4 " free"}')"
fi

echo ""
echo "=== ALL DONE: $(date) ==="
curl -s -H "Title: Pre-quantized Models Done" -H "Priority: high" -H "Tags: checkered_flag" \
  -d "Job $SLURM_JOB_ID: All 3 pre-quantized models finished" \
  https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
