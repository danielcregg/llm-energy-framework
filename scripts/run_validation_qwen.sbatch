#!/bin/bash
#SBATCH --job-name=llm-energy-validation
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Validation run using non-gated model: Qwen2.5-1.5B-Instruct

# Activate conda environment with PyTorch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camorl

export HF_HOME=$HOME/hf_cache
export NTFY_TOPIC="llm-energy-framework-127"
mkdir -p slurm_logs

echo "=== Validation Job $SLURM_JOB_ID ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "==="

cd /home/daniel.cregg/llm-energy-framework

# Record GPU environment
nvidia-smi > environment_gpu.txt
echo "---" >> environment_gpu.txt
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" >> environment_gpu.txt
python3 -c "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); print('NVML:', pynvml.nvmlDeviceGetName(h)); pynvml.nvmlShutdown()" >> environment_gpu.txt

# Send notification
curl -s -H "Title: Validation Started" -H "Tags: hourglass" \
  -d "Validation job $SLURM_JOB_ID started on $(hostname). Model: Qwen2.5-1.5B-Instruct (fp16)" \
  https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1 &

# Run validation benchmark
python3 -m src.benchmark \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --precision fp16 \
    --batch-sizes 1 4 \
    --n-runs 10 \
    --output-dir results/validation

BENCH_EXIT=$?

echo "=== Validation Done: $(date) (exit=$BENCH_EXIT) ==="

# Quick validation checks
echo ""
echo "=== Validation Checks ==="
python3 << 'PYEOF'
import json, glob, sys

files = glob.glob('results/validation/*.json')
if not files:
    print('FAIL: No JSON report found')
    sys.exit(1)

with open(files[0]) as f:
    report = json.load(f)

print(f'Report: {files[0]}')
print(f'Model: {report["model"]["name"]}')
print(f'Baseline: {report["summary"]["idle_baseline_watts"]:.1f} W')

all_ok = True
for r in report['results']:
    j_tok = r['metrics']['joules_per_token']['mean']
    j_std = r['metrics']['joules_per_token']['std']
    cv = j_std / j_tok * 100 if j_tok > 0 else 999
    tok_s = r['metrics']['tokens_per_second']['mean']

    status = 'OK'
    if j_tok < 0.001 or j_tok > 1.0:
        status = 'WARN: J/tok out of expected range [0.001, 1.0]'
        all_ok = False
    elif cv > 15:
        status = f'WARN: CV={cv:.1f}% > 15%'
        all_ok = False

    print(f'  {r["prompt_id"]}/bs={r["batch_size"]}: J/tok={j_tok:.4f} (CV={cv:.1f}%) tok/s={tok_s:.1f} [{status}]')

baseline = report['summary']['idle_baseline_watts']
if baseline < 20 or baseline > 80:
    print(f'WARN: Baseline {baseline:.1f}W outside expected range [20, 80]')
    all_ok = False

if all_ok:
    print('ALL CHECKS PASSED')
else:
    print('SOME CHECKS HAVE WARNINGS - review above')
PYEOF

# Send completion notification
if [ $BENCH_EXIT -eq 0 ]; then
    curl -s -H "Title: Validation Complete" -H "Priority: high" -H "Tags: tada" \
      -d "Validation job $SLURM_JOB_ID completed successfully. Check results/validation/" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
else
    curl -s -H "Title: Validation FAILED" -H "Priority: urgent" -H "Tags: warning" \
      -d "Validation job $SLURM_JOB_ID failed with exit code $BENCH_EXIT. Check slurm_logs/" \
      https://ntfy.sh/$NTFY_TOPIC > /dev/null 2>&1
fi
