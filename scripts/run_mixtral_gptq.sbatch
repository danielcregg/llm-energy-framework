#!/bin/bash
#SBATCH --job-name=llm-energy-mixtral
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Mixtral-8x7B GPTQ-Int4 standalone benchmark (~24GB download)

set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate camorl

export HF_HOME=$HOME/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export NTFY_TOPIC="llm-energy-framework-127"
mkdir -p slurm_logs

cd /home/daniel.cregg/llm-energy-framework

MODEL="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
OUTDIR="results/mixtral-8x7b-gptq"
CACHE_DIR="TheBloke--Mixtral-8x7B-Instruct-v0.1-GPTQ"

echo "=== Job $SLURM_JOB_ID: Mixtral-8x7B GPTQ-Int4 ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi

AVAIL_GB=$(df --output=avail -BG $HOME | tail -1 | tr -d ' G')
echo "Available disk: ${AVAIL_GB}GB"

# Check if model is already cached (no download needed)
if [ -d "$HF_HOME/hub/models--${CACHE_DIR}" ]; then
    echo "Model already cached, skipping disk check"
elif [ "$AVAIL_GB" -lt 30 ]; then
    echo "SKIP: Only ${AVAIL_GB}GB free, need 30GB for Mixtral-8x7B"
    curl -s -H "Tags: warning" -d "Mixtral-8x7B skipped: only ${AVAIL_GB}GB free" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    exit 1
fi

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
echo "=== DONE: $(date) ==="
