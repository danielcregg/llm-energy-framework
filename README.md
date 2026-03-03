# llm-energy-framework

**A Rigorous, Reproducible Framework for Measuring the Energy and Carbon Efficiency of LLM Inference**

Daniel Cregg | University of Galway | Supervisors: Prof. Jim Duggan & Dr. Enda Howley

## Overview

This framework measures the real energy cost of running large language models at inference time. The primary metric is **Joules per Token (J/tok)** — how much energy a GPU consumes to generate each output token, measured directly from hardware using the Nvidia Management Library (NVML).

This is **Contribution 1** (C1) of a PhD programme on carbon-aware LLM benchmarking, certification, and scheduling. The framework and its empirical results will form the basis of the first journal paper.

## Why This Exists

AI energy consumption is growing rapidly, yet public claims about model efficiency are largely self-reported, inconsistent, and unverifiable. There is no independent, standardised benchmark for LLM inference energy efficiency. This framework builds one.

## Architecture

The framework is structured as four layers, each with a clear scope:

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

The framework uses 5 fixed prompts covering different task types for reproducibility:

1. **Summarisation** — Condense a passage into 3 sentences
2. **Question Answering** — Factual recall question
3. **Code Generation** — Write a Python function with error handling
4. **Long-form Generation** — Explain transformer architecture
5. **Reasoning** — Solve a word problem requiring calculation

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

See [docs/output_schema.md](docs/output_schema.md) for the full JSON schema.

## Project Structure

```
llm-energy-framework/
├── src/
│   ├── hardware.py        # Layer 1: NVML power sampling
│   ├── inference.py       # Layer 2: Model loading and inference
│   ├── metrics.py         # Layer 3: Metric computation
│   └── benchmark.py       # Layer 4: Orchestration and reporting
├── prompts/
│   └── benchmark_prompts.json  # Standard prompt set
├── docs/
│   ├── output_schema.md   # JSON report schema documentation
│   └── design_decisions.md # Architecture decisions and rationale
├── results/               # Benchmark output (gitignored)
├── paper/                 # LaTeX paper source
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

## Target Publication

- **Working title:** *llm-energy-framework: A Framework for Reproducible Energy and Carbon Benchmarking of LLM Inference*
- **Target journals:** IEEE Transactions on Sustainable Computing (TSUSC); Future Generation Computer Systems (FGCS)
- **Target submission:** Q1 2027

## Related Repositories

- [phd-plan](https://github.com/danielcregg/phd-plan) — PhD programme planning documents
- [energy-bench](https://github.com/danielcregg/energy-bench) — Prior benchmarking work (A100 GPU, Pythia/Mistral studies)
- [ca-morl](https://github.com/danielcregg/ca-morl) — Carbon-aware multi-objective RL for Kubernetes (C4)
- [ca-morl-gpu](https://github.com/danielcregg/ca-morl-gpu) — GPU-accelerated simulation experiments (C4)

## Licence

MIT
