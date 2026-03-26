#!/bin/bash
# Submit Mixtral batch-sweep jobs: FP16 (per-prompt) and GPTQ (single job)
set -uo pipefail

cd /home/daniel.cregg/llm-energy-framework
mkdir -p slurm_logs

PROMPTS=(sum_short qa_factual code_simple creative_open reasoning)

# --- Mixtral FP16: per-prompt jobs (each ~4h) ---
PREV_JOB=""
for PROMPT in "${PROMPTS[@]}"; do
    DEP_FLAG=""
    if [[ -n "$PREV_JOB" ]]; then
        DEP_FLAG="--dependency=afterany:$PREV_JOB"
    fi

    JOB_ID=$(sbatch --parsable $DEP_FLAG \
        --job-name=llm-energy \
        --partition=defq \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=0 \
        --time=05:00:00 \
        --output=slurm_logs/%j_%x.out \
        --error=slurm_logs/%j_%x.err \
        --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate camorl && \
export HF_HOME=\$HOME/hf_cache && \
python3 -m src.benchmark \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --precision fp16 \
    --batch-sizes 1 2 4 8 16 32 \
    --n-runs 10 \
    --output-dir results/val-mixtral-fp16-batch-sweep/${PROMPT} \
    --prompts ${PROMPT}")

    echo "Mixtral FP16 ${PROMPT}: job ${JOB_ID}"
    PREV_JOB=$JOB_ID
done

# --- Mixtral GPTQ: single job (all prompts, ~5h) ---
GPTQ_JOB=$(sbatch --parsable \
    --dependency=afterany:$PREV_JOB \
    --job-name=llm-energy \
    --partition=defq \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=0 \
    --time=08:00:00 \
    --output=slurm_logs/%j_%x.out \
    --error=slurm_logs/%j_%x.err \
    --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate camorl && \
export HF_HOME=\$HOME/hf_cache && \
python3 -m src.benchmark \
    --model TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ \
    --precision gptq \
    --batch-sizes 1 2 4 8 16 32 \
    --n-runs 10 \
    --output-dir results/val-mixtral-gptq-batch-sweep")

echo "Mixtral GPTQ (all prompts): job ${GPTQ_JOB}"
echo "---"
echo "Total: 5 FP16 jobs + 1 GPTQ job chained sequentially"
