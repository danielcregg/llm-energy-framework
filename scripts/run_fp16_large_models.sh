#!/bin/bash
# Submit FP16 benchmarks for large models previously only run as GPTQ
# Now possible with /HPC storage (98TB) and 2x A100 80GB (160GB VRAM)

set -euo pipefail
cd /home/daniel.cregg/llm-energy-framework
mkdir -p slurm_logs

echo "=== Submitting FP16 large model benchmarks ==="
echo "HF cache: /HPC/daniel.cregg/hf_cache (98TB available)"
echo ""

# --- Job 1: Qwen2.5-32B FP16 (1 GPU, ~64GB VRAM) ---
JOB1=$(sbatch --parsable --time=08:00:00 --gres=gpu:1 \
    scripts/run_benchmark.sbatch \
    Qwen/Qwen2.5-32B-Instruct fp16 results/qwen2.5-32b-fp16 1 4)
echo "Submitted Qwen2.5-32B FP16: Job $JOB1 (1 GPU, bs=1,4)"

# --- Job 2: Mixtral-8x7B FP16 (2 GPUs, ~94GB VRAM) ---
JOB2=$(sbatch --parsable --time=08:00:00 --gres=gpu:2 \
    --dependency=afterany:$JOB1 \
    scripts/run_benchmark.sbatch \
    mistralai/Mixtral-8x7B-Instruct-v0.1 fp16 results/mixtral-8x7b-fp16 1 4)
echo "Submitted Mixtral-8x7B FP16: Job $JOB2 (2 GPUs, bs=1,4, after $JOB1)"

# --- Job 3: Llama-3.3-70B FP16 (2 GPUs, ~140GB VRAM) ---
JOB3=$(sbatch --parsable --time=12:00:00 --gres=gpu:2 \
    --dependency=afterany:$JOB2 \
    scripts/run_benchmark.sbatch \
    meta-llama/Llama-3.3-70B-Instruct fp16 results/llama-3.3-70b-fp16 1)
echo "Submitted Llama-3.3-70B FP16: Job $JOB3 (2 GPUs, bs=1, after $JOB2)"

echo ""
echo "=== All jobs submitted (sequential dependency chain) ==="
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f slurm_logs/<jobid>_llm-energy.out"
