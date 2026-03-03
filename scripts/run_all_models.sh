#!/bin/bash
# Submit all 13 model benchmark jobs via SLURM.
# Run from the project root: bash scripts/run_all_models.sh

set -uo pipefail  # removed -e: don't abort if one sbatch fails

mkdir -p slurm_logs results

SCRIPT="scripts/run_benchmark.sbatch"

echo "=== Submitting LLM Energy Benchmark Jobs ==="
echo "Date: $(date)"
echo ""

# --- Small models (1-4B): 4h time limit ---

echo "1/15: Llama-3.2-1B-Instruct (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    meta-llama/Llama-3.2-1B-Instruct fp16 results/llama-3.2-1b

echo "2/15: Qwen2.5-1.5B-Instruct (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    Qwen/Qwen2.5-1.5B-Instruct fp16 results/qwen2.5-1.5b

echo "3/15: Gemma-2-2b-it (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    google/gemma-2-2b-it fp16 results/gemma-2-2b

echo "4/15: Llama-3.2-3B-Instruct (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    meta-llama/Llama-3.2-3B-Instruct fp16 results/llama-3.2-3b

echo "5/15: Phi-3-mini-4k-instruct (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    microsoft/Phi-3-mini-4k-instruct fp16 results/phi-3-mini

# --- Medium models (7-14B): 4h time limit ---

echo "6/15: Mistral-7B-Instruct-v0.3 (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    mistralai/Mistral-7B-Instruct-v0.3 fp16 results/mistral-7b

echo "7/15: Qwen2.5-7B-Instruct (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    Qwen/Qwen2.5-7B-Instruct fp16 results/qwen2.5-7b

echo "8/15: Llama-3.1-8B-Instruct (fp16) — quantisation study baseline"
sbatch --time=04:00:00 "$SCRIPT" \
    meta-llama/Llama-3.1-8B-Instruct fp16 results/llama-3.1-8b-fp16

echo "9/15: Llama-3.1-8B-Instruct (int8) — quantisation study"
sbatch --time=04:00:00 "$SCRIPT" \
    meta-llama/Llama-3.1-8B-Instruct int8 results/llama-3.1-8b-int8

echo "10/15: Llama-3.1-8B-Instruct (int4) — quantisation study"
sbatch --time=04:00:00 "$SCRIPT" \
    meta-llama/Llama-3.1-8B-Instruct int4 results/llama-3.1-8b-int4

echo "11/15: Gemma-2-9b-it (fp16)"
sbatch --time=04:00:00 "$SCRIPT" \
    google/gemma-2-9b-it fp16 results/gemma-2-9b

echo "12/15: Phi-3-medium-4k-instruct (fp16)"
sbatch --time=06:00:00 "$SCRIPT" \
    microsoft/Phi-3-medium-4k-instruct fp16 results/phi-3-medium

# --- Large models (32-70B): 8h time limit ---

echo "13/15: Qwen2.5-32B-Instruct (int8)"
sbatch --time=08:00:00 "$SCRIPT" \
    Qwen/Qwen2.5-32B-Instruct int8 results/qwen2.5-32b

echo "14/15: Mixtral-8x7B-Instruct-v0.1 (fp16)"
sbatch --time=08:00:00 "$SCRIPT" \
    mistralai/Mixtral-8x7B-Instruct-v0.1 fp16 results/mixtral-8x7b

echo "15/15: Llama-3.3-70B-Instruct (int4)"
sbatch --time=08:00:00 "$SCRIPT" \
    meta-llama/Llama-3.3-70B-Instruct int4 results/llama-3.3-70b

echo ""
echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
