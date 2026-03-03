# Claude Code Autonomous Execution Guide

You are working on the **llm-energy-framework** project — a rigorous framework for measuring LLM inference energy on GPU hardware. Your job is to complete the implementation, run all benchmarks, analyse the results, and produce everything needed for a journal paper.

**Read `RESEARCH_PLAN.md` for the full research plan. This file tells you HOW to execute it.**

## Environment

- **Hardware:** NVIDIA A100 80GB GPU, accessed via SLURM
- **OS:** Linux
- **Python:** 3.10+ (check with `python3 --version`)
- **Access pattern:** You are running on the GPU machine with SLURM job submission

## Critical Rules

1. **Never delete or overwrite existing results.** If re-running a benchmark, save to a timestamped subdirectory.
2. **Commit after every significant milestone** (framework completion, each model benchmarked, analysis done).
3. **If something fails, log the error and move on.** Do not get stuck retrying. Document failures in `results/errors.log`.
4. **Always run validation checks** after each benchmark. If checks fail, investigate before proceeding.
5. **Use greedy decoding only** (`do_sample=False`). Never use sampling, beam search, or temperature.
6. **Use `torch.cuda.synchronize()`** before timing measurements to ensure accurate GPU timing.
7. **Handle OOM gracefully.** If a (model, batch_size) combination causes CUDA OOM, catch the exception, log it, clear GPU memory with `torch.cuda.empty_cache()` and `gc.collect()`, and skip to the next configuration.

## Step-by-Step Execution

### STEP 1: Verify Environment

Run these checks first. Do not proceed until all pass.

```bash
nvidia-smi
python3 --version
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); print('NVML:', pynvml.nvmlDeviceGetName(h)); pynvml.nvmlShutdown()"
```

If any dependency is missing, install it:
```bash
pip install -r requirements.txt
```

Set up the HuggingFace cache:
```bash
export HF_HOME=$HOME/hf_cache
mkdir -p $HF_HOME
```

Record the environment:
```bash
nvidia-smi > environment.txt
echo "---" >> environment.txt
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" >> environment.txt
pip freeze >> environment.txt
```

### STEP 2: Complete `src/hardware.py`

Implement the three TODO sections. Key details:

**SLURM GPU mapping (CRITICAL):**
```python
import os
import pynvml

def _resolve_nvml_index(cuda_index: int = 0) -> int:
    """Map CUDA device index to physical NVML index under SLURM."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return cuda_index
    entries = [e.strip() for e in cvd.split(",") if e.strip()]
    if cuda_index >= len(entries):
        return cuda_index
    try:
        return int(entries[cuda_index])
    except ValueError:
        # UUID format — resolve via NVML
        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                dev_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(dev_uuid, bytes):
                    dev_uuid = dev_uuid.decode()
                if entries[cuda_index] in dev_uuid:
                    return i
        finally:
            pynvml.nvmlShutdown()
    return cuda_index
```

**Idle baseline:** Sample for 30s at 100ms intervals. Compute mean. Check std < 10% of mean. If unstable, extend to 60s.

**Power sampling thread:** Daemon thread that appends `(time.monotonic(), watts)` tuples. `nvmlDeviceGetPowerUsage()` returns milliwatts — divide by 1000.

**Trapezoidal integration:**
```python
energy = sum(
    (samples[i][1] + samples[i-1][1]) / 2 * (samples[i][0] - samples[i-1][0])
    for i in range(1, len(samples))
)
```

### STEP 3: Complete `src/inference.py`

**Model loading:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_name, precision="fp16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer
```

**Inference with precise timing:**
```python
@torch.inference_mode()
def run_inference(model, tokenizer, prompt, ...):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    # Replicate for batch_size > 1
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)
        inputs["attention_mask"] = inputs["attention_mask"].repeat(batch_size, 1)

    input_ids = input_ids.to(model.device)
    inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(
        input_ids,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,  # GREEDY ONLY
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    output_tokens_per_seq = outputs.shape[1] - prompt_len
    total_output_tokens = output_tokens_per_seq * batch_size
    output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return InferenceResult(
        prompt_id=prompt_id, task_type=task_type,
        prompt_tokens=prompt_len * batch_size,
        output_tokens=total_output_tokens,
        generation_time_seconds=t1 - t0,
        output_text=output_text,
        batch_size=batch_size,
    )
```

### STEP 4: Validation Run

Run the smallest model to validate the entire pipeline:

```bash
python3 -m src.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --precision fp16 \
    --batch-sizes 1 4 \
    --n-runs 10 \
    --output-dir results/validation
```

**Check the output:**
- J/tok should be 0.01-0.2 for a 1B model
- Idle baseline should be 30-60W
- CV < 15% for J/tok across runs
- JSON report exists and is valid

If the validation fails, debug and fix before moving on. Do NOT proceed to the full benchmark campaign with a broken framework.

After validation passes, commit everything:
```bash
git add -A
git commit -m "Complete framework implementation and validation

- Implemented hardware.py: NVML power sampling with SLURM GPU mapping
- Implemented inference.py: Model loading with fp16/int8/int4 precision
- Validated on Llama-3.2-1B: J/tok in expected range, CV < 15%
- First benchmark JSON report produced"
```

### STEP 5: Write Unit Tests

Create `tests/` directory with:
- `test_metrics.py` — test `compute_metrics`, `aggregate_runs`, CI math
- `test_hardware.py` — test trapezoidal integration with known power curves
- `test_inference.py` — test token counting logic

Run: `python3 -m pytest tests/ -v`

Commit the tests.

### STEP 6: SLURM Benchmark Scripts

Create `scripts/run_benchmark.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=llm-energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# Usage: sbatch scripts/run_benchmark.sbatch <model_name> <precision> <output_subdir>
MODEL=${1:?Usage: sbatch run_benchmark.sbatch MODEL PRECISION OUTPUT_DIR}
PRECISION=${2:-fp16}
OUTPUT_DIR=${3:-results/$MODEL}

export HF_HOME=$HOME/hf_cache
mkdir -p slurm_logs

echo "=== Job $SLURM_JOB_ID: $MODEL ($PRECISION) ==="
nvidia-smi
echo "==="

python3 -m src.benchmark \
    --model "$MODEL" \
    --precision "$PRECISION" \
    --batch-sizes 1 4 8 16 \
    --n-runs 10 \
    --output-dir "$OUTPUT_DIR"
```

Create `scripts/run_all_models.sh` that submits all 13 models as separate SLURM jobs. Use `--time=08:00:00` for models > 14B.

### STEP 7: Run All Benchmarks

Submit all jobs via `scripts/run_all_models.sh`.

**Model execution order (smallest first so results come in quickly):**
1. Llama-3.2-1B-Instruct (fp16)
2. Qwen2.5-1.5B-Instruct (fp16)
3. Gemma-2-2b-it (fp16)
4. Llama-3.2-3B-Instruct (fp16)
5. Phi-3-mini-4k-instruct (fp16)
6. Mistral-7B-Instruct-v0.3 (fp16)
7. Qwen2.5-7B-Instruct (fp16)
8. Llama-3.1-8B-Instruct (fp16, int8, int4) — quantisation study
9. Gemma-2-9b-it (fp16)
10. Phi-3-medium-4k-instruct (fp16)
11. Qwen2.5-32B-Instruct (int8)
12. Mixtral-8x7B-Instruct-v0.1 (fp16 or int8)
13. Llama-3.3-70B-Instruct (int4)

After each job completes:
- Check the SLURM log for errors
- Verify the JSON report exists and passes validation checks
- If a model failed, check why (OOM? timeout? download issue?) and re-run with adjusted settings
- Commit the results

### STEP 8: Analysis

Create `src/analyze.py` that:

1. **Loads all JSON reports** from `results/` subdirectories
2. **Builds a combined DataFrame** with columns: model, params, precision, prompt_id, batch_size, j_per_tok_mean, j_per_tok_std, tok_per_s_mean, mean_watts, etc.
3. **Exports `results/combined_results.csv`**
4. **Generates all 7 figures** listed in RESEARCH_PLAN.md (save to `paper/figures/`)
5. **Generates the summary table** (save to `paper/tables/`)

Use `matplotlib` for figures. Style requirements:
- Font size: 12pt minimum for axis labels
- Figure size: appropriate for single-column (3.5 inch) or double-column (7 inch) IEEE format
- Save as both PNG (300 DPI for preview) and PDF (for paper)
- Use colourblind-friendly palettes (e.g., matplotlib's "tab10" or seaborn's "colorblind")

Commit all figures and analysis output.

### STEP 9: Paper LaTeX

Create the paper structure in `paper/`:
- `main.tex` — IEEE conference/journal format
- `references.bib` — BibTeX references
- `figures/` — all figures from Step 8
- `tables/` — all tables from Step 8

The paper structure follows Section 5.1 of RESEARCH_PLAN.md.

Generate figures and tables programmatically from the JSON data — do not hard-code numbers.

### STEP 10: Final Commit and Cleanup

- Run all unit tests: `python3 -m pytest tests/ -v`
- Verify all JSON reports are committed
- Verify all figures are committed
- Update README.md with actual results summary
- Final commit

## Handling Problems

**Model download fails:**
- Check disk space (`df -h $HF_HOME`)
- Try `huggingface-cli login` if model is gated (Llama requires access approval)
- If a specific model is unavailable, document it and substitute a similar model

**CUDA OOM:**
- Reduce batch size and retry
- Try int8 or int4 precision for large models
- Log which (model, batch_size, precision) combinations OOM'd — this is useful data for the paper

**SLURM job timeout:**
- Increase `--time` and resubmit
- Consider splitting into per-batch-size jobs for very large models

**Unstable baseline power:**
- Increase baseline measurement to 60 seconds
- Check for background GPU processes: `nvidia-smi`
- If another user's job is sharing the node, request exclusive access: `#SBATCH --exclusive`

**NVML returns 0 watts:**
- This happens on some virtualised/cloud GPUs — check if NVML power reading is supported
- Use `nvidia-smi --query-gpu=power.draw --format=csv` as a cross-check

**Token count mismatch:**
- Some models add special tokens during generation — ensure you subtract only `prompt_len`, not `prompt_len + special_tokens`
- Verify with `tokenizer.decode()` on the output and manually count

## File Hygiene

- **Do not create documentation files** unless specified in the plan
- **Do not refactor working code** unless it's broken
- **Do not add features** beyond what's in the research plan
- **Commit frequently** with clear messages describing what was done
- **Keep results** — never delete a completed benchmark report

## Completion Checklist

You are done when ALL of these are true:
- [ ] `src/hardware.py` — fully implemented, tested
- [ ] `src/inference.py` — fully implemented, tested
- [ ] `src/metrics.py` — tested (already implemented)
- [ ] `src/benchmark.py` — tested end-to-end
- [ ] `src/analyze.py` — analysis and figure generation
- [ ] Validation run passes all checks
- [ ] All 13 models benchmarked (or documented why a model was skipped)
- [ ] Llama-3.1-8B quantisation study complete (fp16, int8, int4)
- [ ] All 7 figures generated in `paper/figures/`
- [ ] Summary table generated in `paper/tables/`
- [ ] Combined CSV at `results/combined_results.csv`
- [ ] Unit tests pass
- [ ] `environment.txt` committed
- [ ] Paper LaTeX skeleton in `paper/`
- [ ] All results committed to git
