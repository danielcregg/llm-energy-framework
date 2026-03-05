# JouleBench

**JouleBench: A Rigorous, Reproducible Framework for Measuring the Energy and Carbon Efficiency of LLM Inference**

Daniel Cregg | South East Technological University | Waterford, Ireland

## Overview

JouleBench measures the real energy cost of running large language models at inference time. The primary metric is **Joules per Token (J/tok)** — how much energy a GPU consumes to generate each output token, measured directly from hardware using the Nvidia Management Library (NVML).

This is **Contribution 1** (C1) of a PhD programme on carbon-aware LLM benchmarking, certification, and scheduling. JouleBench and its empirical results will form the basis of the first journal paper.

## Why This Exists

AI energy consumption is growing rapidly, yet public claims about model efficiency are largely self-reported, inconsistent, and unverifiable. There is no independent, standardised benchmark for LLM inference energy efficiency. JouleBench builds one.

## Architecture

JouleBench is structured as four layers, each with a clear scope:

```
Layer 1: Hardware Instrumentation (src/hardware.py)
  └─ NVML power sampling at 100ms intervals, idle baseline subtraction

Layer 2: Inference Execution Engine (src/inference.py)
  └─ Standardised prompts, precise token counting, warm-up runs, batch size sweeps

Layer 3: Metric Computation (src/metrics.py)
  └─ J/tok, tok/s, mean watts, gCO2eq/tok (with grid carbon intensity)

Layer 4: Orchestration & Reporting (src/benchmark.py)
  └─ Full benchmark runs, statistical aggregation (mean, std, 95% CI), JSON reports
```

## Metrics

| Metric | Unit | Description |
|---|---|---|
| **Joules per Token** | J/tok | Primary metric — net inference energy / output tokens |
| **Tokens per Second** | tok/s | Inference throughput |
| **Mean Power Draw** | W | Average GPU wattage during inference |
| **gCO2eq per Token** | gCO2eq/tok | Carbon cost given grid carbon intensity (optional) |

## Standard Benchmark Prompts

JouleBench uses 5 fixed prompts covering different task types for reproducibility:

1. **Summarisation** — Condense a passage into 3 sentences
2. **Question Answering** — Factual recall question
3. **Code Generation** — Write a Python function with error handling
4. **Long-form Generation** — Explain transformer architecture
5. **Reasoning** — Solve a word problem requiring calculation

## Key Results

Benchmarked on an NVIDIA A100 80GB PCIe GPU. All measurements use direct NVML power sampling with idle baseline subtraction and trapezoidal energy integration.

### Models Benchmarked

| Model | Params (B) | Precision | Best J/tok | Best tok/s | Mean W |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.0 | FP16 | 0.031 | 1284.2 | 100.7 |
| Qwen2.5-1.5B-Instruct | 1.5 | FP16 | 0.040 | 736.3 | 93.7 |
| Gemma-2-2b-it | 2.0 | FP16 | 0.066 | 591.9 | 98.6 |
| Llama-3.2-3B-Instruct | 3.0 | FP16 | 0.083 | 744.4 | 120.9 |
| Phi-3-mini-4k-instruct | 3.8 | FP16 | 0.135 | 665.1 | 155.3 |
| Mistral-7B-Instruct-v0.3 | 7.0 | FP16 | 0.166 | 672.4 | 170.3 |
| Qwen2.5-7B-Instruct | 7.0 | FP16 | 0.165 | 725.5 | 179.4 |
| Llama-3.1-8B-Instruct | 8.0 | FP16 | 0.175 | 663.8 | 176.7 |
| Llama-3.1-8B-Instruct | 8.0 | INT8 | 0.356 | 145.3 | 112.1 |
| Llama-3.1-8B-Instruct | 8.0 | INT4 | 0.944 | 226.4 | 274.2 |
| Gemma-2-9b-it | 9.0 | FP16 | 0.228 | 371.4 | 144.3 |
| Phi-3-medium-4k-instruct | 14.0 | FP16 | 0.334 | 535.8 | 240.4 |
| Qwen2.5-32B-Instruct-GPTQ-Int4 | 32.0 | GPTQ | 10.3 | 23.2 | 298.0 |
| Mixtral-8x7B-Instruct-v0.1-GPTQ | 46.7 | GPTQ | 4.15 | 54.8 | 285.6 |
| Llama-3.3-70B-Instruct-GPTQ | 70.0 | GPTQ | 326.5* | 0.7* | 298.6 |

Best J/tok and tok/s are at optimal batch size (bs=16 unless noted). 295 individual measurements across 16 benchmark configurations. GPTQ models use pre-quantized weights via auto-gptq. *Llama-70B tested at bs=1 only (~60min per config). Mixtral-8x7B is a Mixture-of-Experts model (46.7B total params, ~12.9B active per token), explaining its lower J/tok than the dense Qwen-32B.

### Key Findings

1. **Scaling law**: Energy scales as N^0.91 across architectures (R²=0.99), steeper than the 0.80 exponent found in single-family (Pythia) studies.
2. **Batch size**: Increasing from bs=1 to bs=16 reduces J/tok by 9-15x.
3. **Quantisation reversal**: At bs=1, bitsandbytes INT8 and INT4 on Llama-3.1-8B *reduce* per-token energy by 10% and 20% respectively, reversing the 2.9-3.7x energy penalties reported in prior work. At larger batch sizes, the throughput penalty dominates and FP16 remains more efficient.
4. **Carbon variation**: Irish grid carbon intensity (119-348 gCO2/kWh) introduces 2.9x variation in per-token emissions, meaning *when* you run inference matters as much as *which* model you choose.

### Figures

All figures are in `paper/figures/` (PDF and PNG):
- `fig1_scaling_law` — J/tok vs model parameters with power law fit
- `fig2_efficiency_frontier` — Efficiency frontier across models
- `fig3_batch_size` — J/tok vs batch size for all models
- `fig4_quantisation` — Quantisation impact on Llama-3.1-8B
- `fig5_prompt_type` — Energy variation by prompt type
- `fig6_architecture` — Architecture comparison at similar sizes
- `fig7_carbon_variation` — Carbon emissions under varying grid intensity
- `prior_scaling_overlay` — Comparison with energy-bench Pythia scaling law
- `prior_quantisation_comparison` — Quantisation comparison with prior work
- `prior_batch_saturation` — Batch size saturation comparison

## Quick Start

### Prerequisites

- Nvidia GPU with CUDA
- Python 3.10+
- PyTorch with CUDA support

### Install

```bash
git clone https://github.com/danielcregg/llm-energy-framework.git
cd llm-energy-framework
pip install -r requirements.txt
```

### Run a Benchmark

```bash
python src/benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --precision fp16 \
  --batch-sizes 1 4 \
  --n-runs 10 \
  --output-dir results/
```

### Verify Environment

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetName(h))"
```

## Output

Each benchmark produces a structured JSON report containing:

- Model identity and version
- Full hardware and software environment
- Per-prompt, per-batch-size metrics with mean, std, and 95% confidence intervals
- Summary statistics and best-efficiency configuration
- All raw data needed for independent reproduction

## Project Structure

```
llm-energy-framework/
├── src/
│   ├── hardware.py        # Layer 1: NVML power sampling with SLURM GPU mapping
│   ├── inference.py       # Layer 2: Model loading (FP16/INT8/INT4) and inference
│   ├── metrics.py         # Layer 3: Metric computation (J/tok, tok/s, gCO2eq/tok)
│   ├── benchmark.py       # Layer 4: Orchestration and reporting
│   └── analyze.py         # Analysis, figure, and table generation
├── tests/                 # Unit tests (29 tests)
├── scripts/               # SLURM job submission scripts
├── prompts/
│   └── benchmark_prompts.json  # Standard 5-prompt set
├── prior_work/            # Reference data from energy-bench (808 measurements)
├── data/carbon_cache/     # Irish grid carbon intensity data (EirGrid)
├── results/               # Benchmark JSON reports and combined CSV
├── paper/
│   ├── main.tex           # IEEE conference paper
│   ├── references.bib     # BibTeX references
│   ├── figures/           # All figures (PDF + PNG)
│   └── tables/            # Summary and comparison tables (CSV + LaTeX)
├── requirements.txt
├── .gitignore
└── README.md
```

## Research Questions

1. How can LLM inference energy be measured accurately, reproducibly and independently on real GPU hardware?
2. What is the correct unit of measurement, and how should it be normalised across different model sizes and architectures?
3. How do model size, architecture, quantisation level, context length and batch size each affect Joules per Token?
4. What does the landscape of inference energy efficiency look like across a representative set of publicly available open-weight LLMs?
5. How can measured Joules per Token be combined with grid carbon intensity data to produce a carbon efficiency figure?

## Publication

- **Title:** *JouleBench: Energy Consumption of Large Language Model Inference — A Multi-Architecture Benchmarking Study on GPU Hardware*
- **Target venues:** IEEE Transactions on Sustainable Computing (TSUSC); Future Generation Computer Systems (FGCS)

## Related Repositories

- [phd-plan](https://github.com/danielcregg/phd-plan) — PhD programme planning documents
- [energy-bench](https://github.com/danielcregg/energy-bench) — Prior benchmarking work (A100 GPU, Pythia/Mistral studies)
- [ca-morl](https://github.com/danielcregg/ca-morl) — Carbon-aware multi-objective RL for Kubernetes (C4)
- [ca-morl-gpu](https://github.com/danielcregg/ca-morl-gpu) — GPU-accelerated simulation experiments (C4)

## Licence

MIT
