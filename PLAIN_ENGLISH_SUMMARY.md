# JouleBench — What We Did and What We Found (Plain English)

## The Problem

Every time you ask ChatGPT a question or use an AI assistant, a powerful GPU somewhere is burning electricity to generate the answer. But nobody really knows how much energy different AI models actually use. Companies self-report numbers that are hard to verify, and there's no standard way to measure it. We wanted to fix that.

## What We Built

We built JouleBench, a tool that plugs directly into the GPU's power meter (via Nvidia's NVML library) and measures exactly how many Joules of energy the GPU uses to generate each token (a token is roughly a word or part of a word). Think of it like putting a smart meter on the GPU.

The tool works in four layers:

1. **Power measurement** — A background thread reads the GPU's power draw 10 times per second while the model is running, then calculates total energy using the area under the power curve (trapezoidal integration). We also measure idle power and subtract it, so we only count the energy used by the actual AI work.

2. **Model running** — We load each AI model onto the GPU and give it the same 5 test questions every time (a summarisation task, a factual question, a coding task, a long explanation, and a maths problem). This keeps things fair across models.

3. **Calculating results** — From the raw power readings and timing data, we compute Joules per token (J/tok), tokens per second (throughput), and average power draw. We also convert energy into carbon emissions using real electricity grid data from Ireland.

4. **Running the full experiment** — For each model, we do 3 warm-up runs (thrown away) and 10 real measured runs, then report the average with confidence intervals. We test at batch sizes 1, 4, 8, and 16 (batch size = how many requests the GPU handles simultaneously).

## What We Tested

We tested **11 AI models** ranging from 1 billion to 32 billion parameters across five model families:

- **Llama** (Meta) — 1B, 3B, and 8B versions
- **Qwen** (Alibaba) — 1.5B, 7B, and 32B versions
- **Gemma** (Google) — 2B and 9B versions
- **Phi** (Microsoft) — 3.8B and 14B versions
- **Mistral** (Mistral AI) — 7B

All models ran in full precision (FP16 — every number stored as a 16-bit floating point) on a single **NVIDIA A100 80GB GPU**. Every model fits entirely in GPU memory, so the measurements are uniform and directly comparable — no compression tricks, no CPU offloading, no hidden variables.

We collected **246 individual measurements** across 11 model configurations.

## What We Found

### 1. Bigger models use more energy (roughly proportionally)

Energy per token scales approximately linearly with model size: roughly E = N^1.10, where N is the number of parameters. This means a model with 10x more parameters uses about 12.6x more energy per token. The relationship is very strong (R-squared = 1.00) across five different model families. This is higher than the 0.80 exponent found in prior single-family (Pythia) studies, reflecting the broader architectural diversity in our benchmark.

### 2. Batching is the single biggest efficiency lever

When the GPU processes just one request at a time (batch size 1), most of its capacity is wasted loading model weights. When you batch 16 requests together, the GPU shares that overhead across all of them. The result: **9 to 15 times less energy per token** at batch size 16 compared to batch size 1. This was the single largest efficiency factor we found.

To put this in perspective: a 1B model processing 16 requests at once uses 0.031 J/tok, while a 14B model processing one request at a time uses 4.39 J/tok — that's a **141x difference**.

### 3. Architecture matters at similar sizes

At the 7-9B scale, Mistral-7B and Qwen-7B were the most energy-efficient, while Gemma-9B used about 35% more energy per token despite being a similar size. This means choosing the right model architecture can save significant energy without changing model capability.

### 4. When you run AI matters as much as which model you run

We used real electricity grid data from Ireland (EirGrid, one week in January 2026) to convert energy into carbon emissions. The carbon intensity of Irish electricity varies from 119 to 348 grams of CO2 per kilowatt-hour depending on the time of day (driven by wind generation). That means the same AI query produces **2.9x more carbon** at the worst time compared to the best time. This variation is larger than the energy difference between some model choices, meaning scheduling AI workloads during low-carbon periods can matter more than picking a slightly more efficient model.

## The Numbers at a Glance

| Model | Size | Energy per Token | Speed | GPU Power |
|---|---|---|---|---|
| Llama-3.2-1B | 1B params | 0.031 J/tok | 1,284 tok/s | 101W |
| Qwen2.5-7B | 7B params | 0.165 J/tok | 726 tok/s | 179W |
| Llama-3.1-8B | 8B params | 0.114 J/tok | 1,285 tok/s | 208W |
| Phi-3-medium | 14B params | 0.334 J/tok | 536 tok/s | 240W |
| Qwen2.5-32B | 32B params | 3.051 J/tok | 74.9 tok/s | 291W |

(All numbers at the best batch size tested.)

## What This Means in Practice

- **If you want cheap, fast AI**: Use the smallest model that's good enough for your task, and batch requests together. A 1B model at batch size 16 is absurdly efficient.
- **If you care about carbon**: Schedule AI workloads during times when the electricity grid is running on renewables. In Ireland, this typically means overnight when wind generation peaks.
- **If you're choosing between models of similar size**: Architecture matters. At the 7-9B scale, Mistral-7B and Qwen-7B were the most energy-efficient, while Gemma-9B used about 35% more energy per token despite being a similar size.

## How to Reproduce This

JouleBench is open source. Clone the repository, install the requirements, point it at a GPU, and run:

```bash
python src/benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --precision fp16 \
  --batch-sizes 1 4 8 16 \
  --n-runs 10 \
  --output-dir results/my_test
```

The tool will measure idle power, run warmup iterations, take 10 measured runs with concurrent power sampling, and produce a JSON report with all the raw data and computed metrics.

## Project Status

The benchmarking campaign is complete. All 11 models have been tested in FP16 precision, all figures generated, and the results have been written up as an IEEE-format academic paper. The paper is currently being prepared for journal submission.
