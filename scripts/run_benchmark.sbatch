#!/bin/bash
#SBATCH --job-name=llm-energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Usage: sbatch scripts/run_benchmark.sbatch <model_name> <precision> <output_subdir> [batch_sizes...]
MODEL=${1:?Usage: sbatch run_benchmark.sbatch MODEL PRECISION OUTPUT_DIR [BATCH_SIZES...]}
PRECISION=${2:-fp16}
OUTPUT_DIR=${3:-results/$MODEL}
shift 3 2>/dev/null
BATCH_SIZES="${@:-1 4 8 16}"

export HF_HOME=$HOME/hf_cache
mkdir -p slurm_logs

echo "=== Job $SLURM_JOB_ID: $MODEL ($PRECISION) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "==="

cd /home/daniel.cregg/llm-energy-framework

python3 -m src.benchmark \
    --model "$MODEL" \
    --precision "$PRECISION" \
    --batch-sizes $BATCH_SIZES \
    --n-runs 10 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done: $(date) ==="
