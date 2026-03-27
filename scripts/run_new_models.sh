#!/bin/bash
# Submit 3 new FP16 models for benchmarking
# All fit entirely on A100 80GB in FP16 — no asterisks
# Run sequentially to avoid disk contention on model downloads

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

echo "Submitting 3 new FP16 benchmarks..."

# 1. Qwen2.5-14B-Instruct (28GB weights, ~4h)
JOB1=$(sbatch --parsable --time=06:00:00 scripts/run_benchmark.sbatch \
    "Qwen/Qwen2.5-14B-Instruct" fp16 results/qwen2.5-14b-fp16 \
    1 4 8 16)
echo "Job $JOB1: Qwen2.5-14B-Instruct (fp16)"

# 2. Mistral-Small-24B (44GB weights, ~6h) — after job 1 finishes
JOB2=$(sbatch --parsable --time=08:00:00 --dependency=afterany:$JOB1 scripts/run_benchmark.sbatch \
    "mistralai/Mistral-Small-Instruct-2409" fp16 results/mistral-small-24b-fp16 \
    1 4 8 16)
echo "Job $JOB2: Mistral-Small-24B (fp16) — after $JOB1"

# 3. Gemma-2-27B-it (54GB weights, ~8h) — after job 2 finishes
JOB3=$(sbatch --parsable --time=08:00:00 --dependency=afterany:$JOB2 scripts/run_benchmark.sbatch \
    "google/gemma-2-27b-it" fp16 results/gemma-2-27b-fp16 \
    1 4 8 16)
echo "Job $JOB3: Gemma-2-27B-it (fp16) — after $JOB2"

echo ""
echo "All jobs submitted. Chain: $JOB1 -> $JOB2 -> $JOB3"
echo "Monitor: squeue -u \$USER"
