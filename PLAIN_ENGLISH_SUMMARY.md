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

We tested **13 AI models** ranging from tiny (1 billion parameters) to huge (70 billion parameters) across five model families:

- **Llama** (Meta) — 1B, 3B, 8B, and 70B versions
- **Qwen** (Alibaba) — 1.5B, 7B, and 32B versions
- **Gemma** (Google) — 2B and 9B versions
- **Phi** (Microsoft) — 3.8B and 14B versions
- **Mistral/Mixtral** (Mistral AI) — 7B and a 47B mixture-of-experts model

The smaller models (up to 14B) ran in full precision (FP16 — every number stored as a 16-bit floating point). The three largest models were too big to fit in memory at full precision, so we used pre-compressed (GPTQ-Int4) versions. We also tested one model (Llama 8B) at three different precision levels to see how compression affects energy.

All testing was done on a single **NVIDIA A100 80GB GPU** on a university HPC cluster. We collected **295 individual measurements** across 16 different configurations.

## What We Found

### 1. Bigger models use more energy (but not as much as you'd think)

Energy per token scales with model size following a power law: roughly E = N^0.68, where N is the number of parameters for dense (non-MoE) architectures. This means a model with 10x more parameters uses about 4.8x more energy per token, not 10x. The relationship is strong (R-squared = 0.90) across five different model families. Mixtral-8x7B, a Mixture-of-Experts model with 46.7B total parameters, is a notable outlier---it uses energy comparable to a 12B dense model because only ~12B parameters are active per token.

### 2. Batching is the single biggest efficiency lever

When the GPU processes just one request at a time (batch size 1), most of its capacity is wasted loading model weights. When you batch 16 requests together, the GPU shares that overhead across all of them. The result: **9 to 15 times less energy per token** at batch size 16 compared to batch size 1. This was the single largest efficiency factor we found.

To put this in perspective: a 1B model processing 16 requests at once uses 0.031 J/tok, while a 14B model processing one request at a time uses 4.39 J/tok — that's a **141x difference**.

### 3. Compression now saves energy (it didn't used to)

Previous research (including our own earlier work) found that compressing model weights (quantisation) actually *increased* total energy by 2.9-3.7x. The compression reduced power draw but slowed the model down so much that it used more energy overall.

We found the opposite: on Llama 8B, INT8 compression reduced energy by 10% and INT4 compression reduced it by 20% (at batch size 1). The software libraries (bitsandbytes) have improved enough that the speed penalty is smaller, and the power savings now win out.

**Important caveat**: This only holds at batch size 1. At larger batch sizes, the speed penalty of compression still dominates, and full-precision models remain more energy-efficient.

### 4. When you run AI matters as much as which model you run

We used real electricity grid data from Ireland (EirGrid, one week in January 2026) to convert energy into carbon emissions. The carbon intensity of Irish electricity varies from 119 to 348 grams of CO2 per kilowatt-hour depending on the time of day (driven by wind generation). That means the same AI query produces **2.9x more carbon** at the worst time compared to the best time. This variation is larger than the energy difference between some model choices, meaning scheduling AI workloads during low-carbon periods can matter more than picking a slightly more efficient model.

## The Numbers at a Glance

| Model | Size | Energy per Token | Speed | GPU Power |
|---|---|---|---|---|
| Llama-3.2-1B | 1B params | 0.031 J/tok | 1,284 tok/s | 101W |
| Qwen2.5-7B | 7B params | 0.165 J/tok | 726 tok/s | 179W |
| Llama-3.1-8B | 8B params | 0.175 J/tok | 664 tok/s | 177W |
| Phi-3-medium | 14B params | 0.334 J/tok | 536 tok/s | 240W |
| Mixtral-8x7B (compressed) | 47B params | 4.15 J/tok | 55 tok/s | 286W |
| Llama-3.3-70B (compressed) | 70B params | 326.5 J/tok | 0.7 tok/s | 299W |

(All numbers at the best batch size tested, except 70B which could only run at batch size 1.)

## What This Means in Practice

- **If you want cheap, fast AI**: Use the smallest model that's good enough for your task, and batch requests together. A 1B model at batch size 16 is absurdly efficient.
- **If you care about carbon**: Schedule AI workloads during times when the electricity grid is running on renewables. In Ireland, this typically means overnight when wind generation peaks.
- **If you're compressing models**: At batch size 1 (single-user scenarios), compression now saves both memory and energy. But for high-throughput serving with large batches, full-precision models are still more energy-efficient per token.
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

The benchmarking campaign is complete. All 13 models have been tested, all figures generated, and the results have been written up as an IEEE-format academic paper. The paper is currently being prepared for journal submission.
