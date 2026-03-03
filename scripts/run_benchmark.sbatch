#!/bin/bash
#SBATCH --job-name=llm-energy
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Usage: sbatch scripts/run_benchmark.sbatch <model_name> <precision> <output_subdir> [batch_sizes...]

# Activate conda environment with PyTorch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camorl

MODEL=${1:?Usage: sbatch run_benchmark.sbatch MODEL PRECISION OUTPUT_DIR [BATCH_SIZES...]}
PRECISION=${2:-fp16}
OUTPUT_DIR=${3:-results/$MODEL}
shift 3 2>/dev/null
BATCH_SIZES="${@:-1 4 8 16}"

export HF_HOME=$HOME/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export NTFY_TOPIC="llm-energy-framework-127"
mkdir -p slurm_logs

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

echo "=== Job $SLURM_JOB_ID: $MODEL ($PRECISION) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "==="

# Disk space safety check — need at least 20GB free for model download
AVAIL_GB=$(df --output=avail -BG $HOME | tail -1 | tr -d ' G')
echo "Available disk: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 20 ]; then
    echo "ERROR: Only ${AVAIL_GB}GB free, need at least 20GB. Aborting."
    curl -s -H "Title: DISK FULL" -H "Priority: urgent" -H "Tags: warning" \
      -d "Job $SLURM_JOB_ID: $MODEL_SHORT only ${AVAIL_GB}GB free, need 20GB" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
    exit 1
fi

curl -s -H "Title: Benchmark Started" -H "Tags: hourglass" \
  -d "Job $SLURM_JOB_ID: $MODEL_SHORT ($PRECISION) on $(hostname)" \
  https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1 &

cd /home/daniel.cregg/llm-energy-framework

python3 -m src.benchmark \
    --model "$MODEL" \
    --precision "$PRECISION" \
    --batch-sizes $BATCH_SIZES \
    --n-runs 10 \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo "=== Done: $(date) (exit=$EXIT_CODE) ==="

# Clean up HF cache to free disk space for next model
if [ $EXIT_CODE -eq 0 ]; then
    MODEL_CACHE_DIR=$(echo "$MODEL" | sed 's|/|--|g')
    echo "Cleaning cache: $HF_HOME/hub/models--${MODEL_CACHE_DIR}"
    rm -rf "$HF_HOME/hub/models--${MODEL_CACHE_DIR}"
    echo "Disk after cleanup: $(df -h $HOME | tail -1 | awk '{print $4 " free"}')"

    curl -s -H "Title: Benchmark Complete" -H "Priority: high" -H "Tags: tada" \
      -d "Job $SLURM_JOB_ID: $MODEL_SHORT ($PRECISION) completed successfully" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
else
    curl -s -H "Title: Benchmark FAILED" -H "Priority: urgent" -H "Tags: warning" \
      -d "Job $SLURM_JOB_ID: $MODEL_SHORT ($PRECISION) failed (exit=$EXIT_CODE)" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
fi
