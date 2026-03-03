# Prior Work Reference Data

This project builds on substantial prior research across three related repositories. This document catalogues all reusable data, code patterns, findings, and reference numbers that should inform the new framework's implementation and analysis.

## Source Repositories

| Repo | What it contains | Key reusable assets |
|---|---|---|
| [energy-bench](https://github.com/danielcregg/energy-bench) | 808 GPU energy measurements on A100 | Scaling laws, quantisation data, power logger code, SLURM patterns |
| [ca-morl](https://github.com/danielcregg/ca-morl) | Carbon-aware RL for Kubernetes | Real Irish grid carbon intensity data (EirGrid), carbon API integration |
| [ca-morl-gpu](https://github.com/danielcregg/ca-morl-gpu) | GPU-accelerated simulation | A100 performance characteristics, JAX power model |

---

## 1. Prior Energy Measurements (energy-bench)

### 1.1 Scaling Law — Pythia Family (70M to 12B)

All at batch size 8, context length 256, fp16, on A100 80GB PCIe.

| Model | Params | J/tok | tok/s | Power (W) |
|---|---|---|---|---|
| Pythia-70M | 70M | 0.048 | 1,478 | 77 |
| Pythia-410M | 410M | 0.185 | 451 | 86 |
| Pythia-1B | 1.0B | 0.295 | 351 | 107 |
| Pythia-1.4B | 1.4B | 0.353 | 296 | 109 |
| Pythia-2.8B | 2.8B | 0.471 | 228 | 112 |
| Pythia-6.9B | 6.9B | 0.664 | 330 | 224 |
| Pythia-12B | 11.8B | 1.055 | 246 | 264 |

**Scaling law:** `E(N) = 0.048 * (N / 70M)^0.8` where alpha = 0.8 (R^2 = 0.99)

168x size increase (70M to 12B) produces only 22x energy increase. Sub-linear.

**New framework should:** Verify whether this exponent holds across different model families (Llama, Mistral, Qwen, etc.) and not just the Pythia family.

### 1.2 Quantisation Energy Paradox (Mistral-7B)

#### Transformers + bitsandbytes (on-the-fly dequantisation)

| Precision | tok/s (bs=8) | J/tok (bs=8) | Power (W) | vs FP16 |
|---|---|---|---|---|
| FP16 | 233.8 | 0.718 | 171 | 1.0x |
| INT8 | 51.9 | 2.07 | 108 | 2.88x worse |
| NF4 | 96.5 | 2.67 | 259 | 3.72x worse |

**Mechanism:** INT8 draws 37% less power (108W vs 171W) but runs 4.5x slower. The throughput penalty dominates, making total energy worse.

#### vLLM with pre-quantised kernels (AWQ/GPTQ)

| Precision | tok/s (bs=8) | J/tok (bs=8) | Power (W) | vs FP16 |
|---|---|---|---|---|
| vLLM FP16 | 606.5 | 0.39 | 244 | baseline |
| vLLM AWQ | 697.1 | 0.28 | 209 | 28% better |
| vLLM GPTQ | 698.9 | 0.33 | 252 | 15% better |

**Cross-framework gap:** 8.2x between bitsandbytes INT8 (2.07 J/tok) and vLLM AWQ (0.28 J/tok) for the same model.

**New framework should:** Test the same bitsandbytes quantisation (int8, int4) used in energy-bench to confirm whether the paradox reproduces, and document the framework version differences.

### 1.3 Batch Size Saturation (Pythia-6.9B, ctx=256)

| Batch Size | tok/s | J/tok | Power (W) | Efficiency vs bs=1 |
|---|---|---|---|---|
| 1 | 44.7 | 3.72 | 172 | 1.0x |
| 2 | 89.4 | 1.92 | 172 | 1.9x |
| 4 | 176.3 | 1.09 | 195 | 3.4x |
| 8 | 343.3 | 0.65 | 230 | 5.7x |
| 16 | 542.8 | 0.46 | 255 | 8.1x |
| 32 | 736.5 | 0.37 | 276 | 10.1x |
| 64 | 858.5 | 0.34 | 290 | 11.0x |
| 128 | 959.7 | 0.31 | 299 | 12.0x |

**Sweet spot:** 80% of max efficiency (0.31 J/tok) achieved at batch size 32 (0.37 J/tok).

**Diminishing returns curve:**
- bs=1 to 2: 1.93x improvement
- bs=2 to 4: 1.77x
- bs=4 to 8: 1.67x
- bs=8 to 16: 1.41x
- bs=16 to 32: 1.25x
- bs=32 to 64: 1.09x (saturating)
- bs=64 to 128: 1.08x (saturated)

**New framework should:** Test batch sizes 1, 4, 8, 16 for all models (matching energy-bench range), and extended set (32, 64) for one model to confirm saturation point.

### 1.4 Context Length Scaling (Mistral-7B vs Pythia-6.9B)

| Context | Mistral J/tok (bs=1) | Pythia J/tok (bs=8) |
|---|---|---|
| 256 | 4.37 | 0.664 |
| 2048 | 5.54 (1.27x) | 2.264 (3.41x) |
| 4096 | 7.48 (1.71x) | 3.39 (5.1x) |

**Key finding:** Mistral's sliding window attention produces near-linear scaling (1.71x for 16x context), while Pythia's full attention scales much steeper (5.1x). This is an architectural effect worth confirming on newer models.

### 1.5 Phase Separation (Prefill vs Decode, Mistral-7B, bs=1)

| Precision | Prefill J/tok | Decode J/tok | Decode Ratio vs FP16 |
|---|---|---|---|
| FP16 | 0.013 | 1.664 | 1.0x |
| INT8 | 0.095 (7.5x) | 4.954 (3.0x) | 3.0x |
| NF4 | 0.038 (3.0x) | 2.486 (1.5x) | 1.5x |

**Decode dominates:** 98%+ of total energy at bs=1. The 7.5x prefill penalty for INT8 is dramatic but irrelevant because prefill is <2% of total energy.

### 1.6 A100 Hardware Baselines

| Property | Value |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| TDP | 300W |
| Idle power | ~47W |
| Max observed power | 299W (at bs=128) |
| CUDA version | 12.2.2 |
| Driver | 530.30.02 |
| DVFS behaviour | Stays at max clock (1410 MHz) even at low utilisation |

### 1.7 Methodology from energy-bench

| Parameter | energy-bench value | New framework value |
|---|---|---|
| Power sampling rate | 10 Hz (100ms) | 10 Hz (100ms) — same |
| Energy integration | Trapezoidal rule | Trapezoidal rule — same |
| Warmup runs | 3 | 3 — same |
| Measured runs | 5 | 10 — increased for tighter CIs |
| Generated tokens | 128 | 200 — increased for longer runs |
| Decoding | Greedy | Greedy — same |
| Baseline subtraction | Not applied | Applied — key improvement |
| Prompts | 10 science topics | 5 task-diverse prompts — more focused |

---

## 2. Irish Grid Carbon Intensity Data (ca-morl)

### 2.1 Dataset

**File:** `data/carbon_cache/roi_sample_7day.csv`

- **Source:** EirGrid Smart Grid Dashboard API
- **Region:** Republic of Ireland (ROI)
- **Date range:** January 15-21, 2026 (7 days)
- **Resolution:** 15-minute intervals (672 data points)
- **Format:** CSV with columns `timestamp` (ISO) and `gco2_per_kwh` (float)

### 2.2 Statistics

| Metric | Value |
|---|---|
| Minimum | 119.0 gCO2/kWh |
| Maximum | 348.1 gCO2/kWh |
| Mean | ~230 gCO2/kWh |
| Diurnal pattern | Overnight lows (119-145), morning ramp (150-200), midday plateau (190-220), evening peak (260-350) |

**The 2.9x variation (119 to 348)** means the same LLM inference can have 2.9x different carbon impact depending on when it runs. This is the core motivation for carbon-aware scheduling (Paper 3) and should be highlighted in the Discussion section of Paper 1.

### 2.3 EirGrid API Details

- **Endpoint:** `GET https://www.smartgriddashboard.com/DashboardService.svc/data`
- **Parameters:** `area=co2intensity`, `region=ROI`, `datefrom`, `dateto`
- **Response:** JSON with `{"Rows": [{"EffectiveTime": "...", "CO2Intensity": 215.3}]}`
- **Cadence:** ~15-minute updates
- **Fallback:** 200.0 gCO2/kWh (Irish grid annual average)

### 2.4 Usage in New Framework

The carbon data enables computing **gCO2eq per token**:

```
gCO2eq_per_token = (J_per_token / 3,600,000) * carbon_intensity_gco2_per_kwh
```

Example at mean Irish grid (230 gCO2/kWh):
- Llama-3.1-8B at 0.1 J/tok → 0.0000064 gCO2eq/tok
- Mistral-7B INT8 at 2.07 J/tok → 0.000132 gCO2eq/tok

The offline CSV can be used to show how the same model's carbon footprint varies across the day without needing live API access.

---

## 3. A100 GPU Characteristics (ca-morl-gpu)

### 3.1 Throughput Scaling on A100

The JAX simulator achieved perfect linear scaling:

| Parallel Envs | Steps/sec | Per-env throughput |
|---|---|---|
| 1 | 2,414 | 2,414 |
| 128 | 368,730 | 2,881 |
| 1,024 | 2,950,000 | 2,881 |
| 32,768 | 92,500,000 | 2,823 |

**Implication for LLM benchmarking:** The A100 has enormous parallel capacity. If batch size experiments don't saturate the GPU (i.e., power stays below TDP), there may be energy efficiency gains available that our benchmark should capture.

### 3.2 Power Model (Cubic V/f Scaling)

The ca-morl-gpu project confirmed that GPU power follows:

```
P = static_power + dynamic_power * frequency^3 * utilisation
```

This matches the A100's observed behaviour: at bs=1 the GPU draws ~136W (well below 300W TDP), while at bs=128 it draws ~299W (approaching TDP). The cubic relationship means DVFS could theoretically save significant energy at low utilisation, but the A100 stays at max clock (1410 MHz) even when underutilised.

---

## 4. Files Included in `prior_work/`

| File | Source | Description |
|---|---|---|
| `energy_bench_power_logger.py` | energy-bench | Battle-tested NVML power logger with SLURM GPU mapping, phase markers, trapezoidal integration |
| `energy_bench_prompts.json` | energy-bench | Original 10-prompt set (science topics) |
| `energy_bench_pythia_combined.csv` | energy-bench | 280 rows: Pythia scaling law data (70M-12B, bs 1-128) |
| `energy_bench_quant_fp16.csv` | energy-bench | 32 rows: Mistral-7B FP16 baseline |
| `energy_bench_quant_int8.csv` | energy-bench | 32 rows: Mistral-7B INT8 quantisation |
| `energy_bench_quant_nf4.csv` | energy-bench | 32 rows: Mistral-7B NF4 quantisation |
| `energy_bench_batch_saturation.csv` | energy-bench | 40 rows: Pythia-6.9B batch size sweep (1-128) |
| `energy_bench_mistral_7b_standard.csv` | energy-bench | 91 rows: Mistral-7B full benchmark (bs 1-32, ctx 256-4096) |

## 5. Files Included in `data/carbon_cache/`

| File | Source | Description |
|---|---|---|
| `roi_sample_7day.csv` | ca-morl | 672 rows: Irish grid carbon intensity, Jan 15-21 2026, 15-min resolution |

---

## 6. Key Comparisons the New Framework Should Make

### 6.1 Scaling Law Validation

energy-bench found alpha = 0.8 using the Pythia family (same architecture, different sizes). The new framework tests across families (Llama, Mistral, Qwen, Phi, Gemma). Questions:

- Does alpha = 0.8 hold across architectures, or was it Pythia-specific?
- Do MoE models (Mixtral) follow the same scaling, using total params or active params?
- Is the R^2 as clean when mixing architectures?

### 6.2 Quantisation Paradox Confirmation

energy-bench found bitsandbytes INT8 makes energy 2.7-2.9x worse. The new framework should confirm:

- Does this hold on newer transformer versions?
- Does INT4 (NF4) behave differently from INT8?
- Is the mechanism the same (throughput penalty > power reduction)?

### 6.3 Batch Saturation Across Model Sizes

energy-bench found 80% efficiency at bs=32 for Pythia-6.9B. Questions:

- Is the saturation point the same for 1B vs 7B vs 70B models?
- Does quantisation shift the saturation point?

### 6.4 Architecture-Controlled Comparison

energy-bench compared Mistral (sliding window) vs Pythia (full attention) but at different sizes (7B vs 6.9B). The new framework can compare at matched sizes:

- Llama-3.1-8B vs Mistral-7B vs Qwen2.5-7B vs Gemma-2-9b (all 7-9B, different architectures)

### 6.5 Carbon Variation

Using the Irish grid data, compute gCO2eq/tok for all models at:
- Minimum grid intensity (119 gCO2/kWh — overnight wind)
- Mean grid intensity (230 gCO2/kWh — average)
- Maximum grid intensity (348 gCO2/kWh — evening peak)

Show the range of carbon cost for the same model depending on when it runs. This motivates Paper 3 (EirGrid integration) and Paper 4 (carbon-aware scheduling).
