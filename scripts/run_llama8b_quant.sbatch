#!/bin/bash
#SBATCH --job-name=llm-energy-quant
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Llama-3.1-8B quantisation study: fp16, int8, int4 in one job
# Downloads model once, runs all 3 precisions, then cleans cache.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate camorl

export HF_HOME=$HOME/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export NTFY_TOPIC="llm-energy-framework-127"
mkdir -p slurm_logs

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT="Llama-3.1-8B-Instruct"

echo "=== Job $SLURM_JOB_ID: $MODEL (quant study: fp16, int8, int4) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "==="

AVAIL_GB=$(df --output=avail -BG $HOME | tail -1 | tr -d ' G')
echo "Available disk: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 20 ]; then
    echo "ERROR: Only ${AVAIL_GB}GB free. Aborting."
    exit 1
fi

cd /home/daniel.cregg/llm-energy-framework

FAIL=0

for PREC in fp16 int8 int4; do
    echo ""
    echo "=== Quantisation: $PREC ($(date)) ==="
    python3 -m src.benchmark \
        --model "$MODEL" \
        --precision "$PREC" \
        --batch-sizes 1 4 8 16 \
        --n-runs 10 \
        --output-dir "results/llama-3.1-8b-${PREC}"

    EC=$?
    if [ $EC -ne 0 ]; then
        echo "FAILED: $PREC (exit=$EC)"
        FAIL=1
    else
        echo "OK: $PREC"
    fi
done

# Clean cache
MODEL_CACHE_DIR=$(echo "$MODEL" | sed 's|/|--|g')
rm -rf "$HF_HOME/hub/models--${MODEL_CACHE_DIR}"
echo "Cache cleaned. Disk: $(df -h $HOME | tail -1 | awk '{print $4 " free"}')"

echo "=== Done: $(date) ==="

if [ $FAIL -eq 0 ]; then
    curl -s -H "Title: Quant Study Complete" -H "Priority: high" -H "Tags: tada" \
      -d "Job $SLURM_JOB_ID: $MODEL_SHORT fp16/int8/int4 all completed" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
else
    curl -s -H "Title: Quant Study Partial" -H "Priority: urgent" -H "Tags: warning" \
      -d "Job $SLURM_JOB_ID: $MODEL_SHORT quant study had failures" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
fi
