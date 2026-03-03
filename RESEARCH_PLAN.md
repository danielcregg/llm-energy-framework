# Research Plan: LLM Energy Benchmarking Framework

## Goal

Build a rigorous, reproducible framework for measuring the energy consumption of LLM inference on GPU hardware, benchmark 10-15 open-weight models, analyse the results, and produce a journal-quality paper.

**Primary metric:** Joules per Token (J/tok)
**Hardware:** NVIDIA A100 80GB via SLURM
**Target output:** Journal paper for IEEE TSUSC or FGCS, submission Q1 2027

---

## Phase 1: Environment Setup and Framework Completion

**Objective:** Get the framework running end-to-end on the A100.

### 1.1 Environment Verification
- Verify GPU access: `nvidia-smi` shows A100 80GB
- Verify CUDA + PyTorch: `torch.cuda.is_available()` returns True
- Verify NVML access: `pynvml.nvmlInit()` succeeds
- Check Python version (need 3.10+)
- Install dependencies from `requirements.txt`
- Check available disk space for model downloads (need ~200GB for all models)
- Set up HuggingFace cache directory: `export HF_HOME=$HOME/hf_cache`

### 1.2 Implement `hardware.py` (Layer 1)
The stub exists with dataclasses and function signatures. Complete the implementation:

**`measure_idle_baseline()`:**
- Init NVML, get device handle
- Handle SLURM's `CUDA_VISIBLE_DEVICES` mapping (see `resolve_nvml_gpu_index` pattern from energy-bench repo — SLURM assigns a physical GPU that may not be NVML index 0)
- Sample power at 100ms intervals for 30 seconds
- Compute mean, verify stability (std < 10% of mean; if not, extend to 60s)
- Return mean idle watts

**`PowerSampler.__enter__()` / `__exit__()`:**
- Start a daemon background thread that samples `nvmlDeviceGetPowerUsage()` every 100ms
- Store `(time.monotonic(), watts)` tuples in `self._samples`
- On exit: set `self._running = False`, join thread

**`PowerSampler.get_results()`:**
- Compute total energy via trapezoidal integration: `E = sum((P_i + P_{i-1})/2 * (t_i - t_{i-1}))`
- Compute net energy: `total_energy - (baseline_watts * duration)`
- Return `EnergyMeasurement` with all fields populated

**Critical SLURM consideration:** When SLURM assigns GPU 3 via `CUDA_VISIBLE_DEVICES=3`, CUDA sees it as device 0, but NVML still indexes it as device 3. Must resolve the physical NVML index.

### 1.3 Implement `inference.py` (Layer 2)
The stub exists with function signatures. Complete the implementation:

**`load_model()`:**
- Use `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()`
- fp16: `torch_dtype=torch.float16, device_map="auto"`
- int8: `load_in_8bit=True` via `BitsAndBytesConfig`
- int4: `load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16` via `BitsAndBytesConfig`
- Set `model.eval()` and `torch.inference_mode()`
- Handle models that need `trust_remote_code=True`
- Set pad token = eos token if not set (common for decoder-only models)

**`run_inference()`:**
- Tokenize input with `tokenizer(prompt, return_tensors="pt", padding=True)`
- Replicate to batch_size if > 1
- Record `input_token_count = input_ids.shape[1]`
- Generate with `model.generate(do_sample=False, max_new_tokens=max_new_tokens)` (greedy decoding)
- Time the generation with `torch.cuda.synchronize()` before start/stop for accurate timing
- Count output tokens: `generated.shape[1] - input_ids.shape[1]` per sequence, times batch_size
- Decode first sequence for `output_text`
- Return `InferenceResult`

### 1.4 Validation Run
- Run the full pipeline against `meta-llama/Llama-3.2-1B-Instruct` (or smallest available Llama)
- Precision: fp16, batch sizes: 1 and 4, n_runs: 10
- **Sanity checks:**
  - J/tok should be 0.01-0.2 for a 1B model on A100
  - Baseline watts should be 30-60W (A100 idle range)
  - CV (std/mean) for J/tok < 15% across 10 runs
  - Manually verify token count on one run
- If checks fail, debug and fix before proceeding
- Commit the validated framework code and the first JSON report

### 1.5 Unit Tests
- Write tests for `metrics.py` (already fully implemented — test `compute_metrics`, `aggregate_runs`, CI calculation)
- Write tests for `hardware.py` (mock NVML, test trapezoidal integration math)
- Write tests for `inference.py` (mock model, test token counting logic)
- Write a smoke test that runs the full pipeline on a tiny model (e.g., `gpt2`)

---

## Phase 2: Model Selection and Full Benchmark Campaign

**Objective:** Benchmark 10-15 models covering the research questions.

### 2.1 Model Selection

The models must cover three axes: **scale** (1B to 70B), **architecture** (different attention mechanisms), and **quantisation** (fp16, int8, int4 of the same base model).

**Proposed model set (adjust based on what fits in 80GB A100 memory):**

| # | Model | Params | Family | Why |
|---|---|---|---|---|
| 1 | `meta-llama/Llama-3.2-1B-Instruct` | 1B | Llama 3.2 | Smallest Llama, validation model |
| 2 | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Llama 3.2 | Small-mid scale |
| 3 | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Llama 3.1 | Standard mid-range |
| 4 | `meta-llama/Llama-3.3-70B-Instruct` | 70B | Llama 3.3 | Large (int4 only on A100) |
| 5 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Mistral | Sliding window attention |
| 6 | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B (12B active) | Mixtral | MoE architecture |
| 7 | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Qwen 2.5 | Non-Western architecture family |
| 8 | `Qwen/Qwen2.5-7B-Instruct` | 7B | Qwen 2.5 | Direct comparison with Mistral/Llama 8B |
| 9 | `Qwen/Qwen2.5-32B-Instruct` | 32B | Qwen 2.5 | Large dense model (int8 or int4) |
| 10 | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Phi-3 | Small, efficient architecture |
| 11 | `microsoft/Phi-3-medium-4k-instruct` | 14B | Phi-3 | Mid-range, different design |
| 12 | `google/gemma-2-2b-it` | 2B | Gemma 2 | Google architecture |
| 13 | `google/gemma-2-9b-it` | 9B | Gemma 2 | Mid-range Google |

**Quantisation study (on Llama-3.1-8B as the reference model):**
- fp16 baseline
- int8 (bitsandbytes)
- int4 (bitsandbytes NF4)
- Compare all three at batch sizes 1, 4, 8, 16

### 2.2 Benchmark Execution Plan

**Per model, run the following configurations:**
- Precision: fp16 (and int8/int4 for quantisation study models)
- Batch sizes: 1, 4, 8, 16
- All 5 standard prompts
- 10 measured runs per configuration (+ 3 warmup, discarded)
- Max new tokens: 200

**Per model, this produces:**
- 5 prompts x 4 batch sizes x 10 runs = 200 measured inference runs
- Plus 5 x 4 x 3 = 60 warmup runs (discarded)
- One JSON report file per model

**Total across 13 models:** ~2,600 measured runs + ~780 warmup runs

**SLURM execution strategy:**
- Create a SLURM batch script (`scripts/run_benchmark.sbatch`) requesting 1 GPU, appropriate time limit
- Run one model per SLURM job to avoid timeout issues
- Set `--time=04:00:00` for models up to 14B, `--time=08:00:00` for larger models
- Create a master script (`scripts/run_all_models.sh`) that submits all jobs
- Save all output to `results/<model_name>/`
- Log SLURM job IDs for tracking

**Memory constraints (A100 80GB):**
- Models up to ~30B fit in fp16
- 70B models need int4 quantisation (or int8 with careful memory management)
- If a model OOMs at a given batch size, log the error gracefully and continue to the next configuration

### 2.3 Benchmark Execution

Run all models. After each model completes:
- Verify the JSON report was produced
- Run validation checks (J/tok range, CV, baseline stability)
- Commit the JSON report to the repo

---

## Phase 3: Analysis and Visualisation

**Objective:** Analyse the benchmark data and produce publication-quality figures.

### 3.1 Analysis Script (`src/analyze.py`)

Create a comprehensive analysis module that:
- Loads all JSON reports from `results/`
- Produces a combined DataFrame for cross-model comparison
- Generates all figures listed below
- Exports a combined CSV (`results/combined_results.csv`) for the paper

### 3.2 Required Figures (for the paper)

**Figure 1: J/tok vs Model Size (Scaling Law)**
- Log-log scatter plot: x = parameter count, y = mean J/tok
- One point per model (at batch size 1, fp16)
- Fit power law: J/tok = a * N^b (report R^2)
- Colour-code by model family (Llama, Mistral, Qwen, Phi, Gemma)
- This directly answers RQ3 (how model size affects energy)

**Figure 2: J/tok vs Throughput (Efficiency Frontier)**
- Scatter plot: x = tok/s, y = J/tok
- One point per (model, batch_size) configuration
- Pareto frontier line showing the best achievable trade-off
- Annotate key models on the frontier
- This directly answers RQ4 (energy efficiency landscape)

**Figure 3: Batch Size Effect**
- Line plot: x = batch size (1, 4, 8, 16), y = J/tok
- One line per model (or per model family)
- Show diminishing returns and identify the sweet spot
- Secondary y-axis: tok/s to show throughput scaling
- This directly answers RQ3 (batch size effect)

**Figure 4: Quantisation Impact**
- Grouped bar chart: fp16 vs int8 vs int4 for Llama-3.1-8B
- Bars showing J/tok, tok/s, and mean watts
- Show that quantisation may increase or decrease J/tok depending on implementation
- Reference the quantisation paradox finding from energy-bench
- This directly answers RQ3 (quantisation effect)

**Figure 5: Prompt Type Effect**
- Heatmap or grouped bar: rows = models, columns = prompt types
- Cell value = mean J/tok
- Show whether some prompt types are systematically more energy-expensive
- Identify if code generation or reasoning are notably different from QA

**Figure 6: Architecture Comparison**
- Bar chart comparing J/tok across architectures at similar parameter counts:
  - 7-9B: Llama-3.1-8B vs Mistral-7B vs Qwen2.5-7B vs Gemma-2-9b
  - This isolates architectural effects from scale effects
- Note Mistral's sliding window attention vs others' full attention

**Figure 7: Carbon Variation (if grid data available)**
- Line plot showing how gCO2eq/tok for a fixed model varies with grid carbon intensity
- Use representative Irish grid values (100-500 gCO2/kWh range)
- This addresses RQ5 (carbon efficiency grounding)

### 3.3 Summary Statistics

Produce a table for the paper:

| Model | Params | Precision | Best J/tok | Best tok/s | Best Batch | Mean Watts |
|---|---|---|---|---|---|---|

---

## Phase 4: Reproducibility and Documentation

**Objective:** Ensure independent reproducibility.

### 4.1 Reproducibility Package
- All JSON reports committed to `results/`
- `scripts/run_benchmark.sbatch` — exact SLURM job script used
- `scripts/run_all_models.sh` — master execution script
- `environment.txt` — full `pip freeze`, `nvidia-smi`, CUDA version, OS details
- Random seeds and deterministic settings documented

### 4.2 Comparison with Prior Work
- Compare scaling law exponent with energy-bench results (E proportional to N^0.8)
- Compare J/tok ranges with Samsi et al. (2023) and Argerich & Patino-Martinez (2024)
- Compare quantisation findings with energy-bench quantisation paradox
- Document any discrepancies and explain (different hardware, framework version, etc.)

### 4.3 Framework Documentation
- Update `docs/design_decisions.md` with any new decisions made during implementation
- Ensure all functions have docstrings and type hints
- Write `docs/adding_models.md` — guide for benchmarking additional models

---

## Phase 5: Paper Writing Support

**Objective:** Produce all data, tables, and figures needed for the paper.

### 5.1 Paper Structure (from Document 2)
1. Introduction — the LLM energy measurement problem
2. Background and Related Work — existing tools, limitations, gap analysis
3. Framework Design — the four-layer architecture
4. Implementation — energy-bench tooling, hardware, software
5. Validation — reproducibility study
6. Empirical Benchmark Study — results across the model set
7. Discussion — findings, limitations, implications
8. Conclusion and Future Work

### 5.2 Paper Deliverables
- LaTeX source in `paper/` directory
- All figures as PDF/PNG in `paper/figures/`
- Results tables generated programmatically from JSON reports
- BibTeX file with all references

---

## Estimated SLURM Time Budget

| Phase | GPU Hours | Notes |
|---|---|---|
| 1.4 Validation run | ~0.5h | Single small model |
| 2.3 Small models (1-4B, 5 models) | ~10h | ~2h each |
| 2.3 Medium models (7-14B, 6 models) | ~24h | ~4h each |
| 2.3 Large models (32-70B, 2 models) | ~16h | ~8h each |
| 3.x Re-runs if needed | ~5h | Buffer for failures |
| **Total** | **~55h** | |

---

## Success Criteria

The project is complete when:
1. All 13 models benchmarked with valid JSON reports
2. All 7 figures generated and publication-ready
3. All validation checks pass (J/tok range, CV < 15%, baseline stable)
4. Combined CSV and summary table produced
5. Framework code is clean, tested, and documented
6. All results committed to the repo with full reproducibility metadata
