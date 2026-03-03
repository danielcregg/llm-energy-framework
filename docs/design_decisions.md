# Design Decisions

This document records key architectural and methodological decisions.

## 1. NVML-Only Power Measurement

**Decision:** Use NVML (via pynvml) as the sole power measurement source. Do not use estimation-based tools like codecarbon for the primary metric.

**Rationale:** The framework's value proposition is *real, hardware-level* energy measurement. Estimation tools use TDP-based models that can be off by 2-3x. NVML reports actual power draw from the GPU's onboard sensor.

**Trade-off:** NVML only measures GPU power. CPU, DRAM, and system-level power are not captured. These are reported separately for transparency but not included in the primary J/tok metric.

## 2. Idle Baseline Subtraction

**Decision:** Measure idle GPU power for 30 seconds before each benchmark run and subtract it from total inference energy to report *net* inference energy.

**Rationale:** A GPU draws significant idle power (30-60W on an A100) even when doing nothing. Without subtraction, idle power dominates J/tok for fast models, making the metric less meaningful as a measure of inference work.

## 3. Fixed Prompt Set

**Decision:** Use exactly 5 standardised prompts covering summarisation, QA, code generation, long-form generation, and reasoning.

**Rationale:** Reproducibility requires identical inputs across all models. The 5 prompts cover the major LLM use cases and produce varying output lengths, testing both short and long generation.

## 4. Statistical Rigour

**Decision:** Run each configuration at least 10 times and report mean, standard deviation, and 95% confidence intervals.

**Rationale:** GPU power draw has natural variance from thermal effects, DVFS, and background processes. A single measurement is unreliable. 10 runs with CI reporting provides confidence in the results.

## 5. Greedy Decoding

**Decision:** Use greedy decoding (no sampling, no beam search) for all benchmark runs.

**Rationale:** Greedy decoding is deterministic — same input produces same output every time. This isolates energy measurement from the randomness of sampling-based generation.

## 6. Warm-Up Runs

**Decision:** Discard the first 3 runs of each configuration before measurement.

**Rationale:** Initial runs incur one-time costs: CUDA kernel compilation, memory allocation, cache warming. These are not representative of steady-state inference.

## 7. JSON Output Format

**Decision:** Output structured JSON reports (not just CSV) with full environment metadata.

**Rationale:** The reports feed directly into Paper 2's certification framework. Structured JSON with hardware/software provenance supports automated certificate generation and independent verification.
