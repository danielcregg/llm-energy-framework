# Output Schema

Each benchmark run produces a structured JSON report. This document defines the schema.

## Top-Level Structure

```json
{
  "report_id": "uuid4",
  "generated_at": "ISO8601 timestamp",
  "framework_version": "0.1.0",
  "model": { ... },
  "hardware": { ... },
  "software": { ... },
  "benchmark_config": { ... },
  "results": [ ... ],
  "summary": { ... }
}
```

## Model

```json
{
  "name": "meta-llama/Llama-3.2-1B-Instruct",
  "precision": "fp16",
  "parameter_count_billions": 1.0
}
```

## Hardware

```json
{
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_memory_gb": 80,
  "cuda_version": "12.4",
  "driver_version": "550.54.15",
  "cpu_model": "AMD EPYC 7763",
  "ram_gb": 512
}
```

## Software

```json
{
  "python_version": "3.11.8",
  "torch_version": "2.3.0",
  "transformers_version": "4.44.0",
  "pynvml_version": "12.535.161",
  "os": "Ubuntu 22.04"
}
```

## Benchmark Config

```json
{
  "n_runs_per_config": 10,
  "warmup_runs": 3,
  "max_new_tokens": 200,
  "batch_sizes_tested": [1, 4, 8],
  "sampling_interval_ms": 100,
  "idle_baseline_duration_seconds": 30
}
```

## Results (per prompt, per batch size)

```json
{
  "prompt_id": "code_simple",
  "task_type": "code",
  "batch_size": 1,
  "runs": 10,
  "metrics": {
    "joules_per_token": { "mean": 0.042, "std": 0.003, "ci_95_lower": 0.039, "ci_95_upper": 0.045 },
    "tokens_per_second": { "mean": 45.2, "std": 2.1, "ci_95_lower": 43.1, "ci_95_upper": 47.3 },
    "mean_watts": { "mean": 180.4, "std": 12.3, "ci_95_lower": 174.0, "ci_95_upper": 186.8 },
    "output_tokens": { "mean": 142, "std": 8 },
    "input_tokens": { "mean": 48, "std": 0 }
  }
}
```

All metrics include mean, standard deviation, and 95% confidence interval computed across `n_runs` repetitions.

## Summary

```json
{
  "overall_mean_joules_per_token": 0.044,
  "overall_mean_tokens_per_second": 43.7,
  "best_energy_efficiency_config": "batch_size=4",
  "idle_baseline_watts": 45.2
}
```

## Validation Checks

After producing a report, verify:

1. **J/tok sanity**: For a 1B model on a high-end GPU, expect 0.01-0.2 J/tok. Values outside this range suggest a measurement bug.
2. **Baseline stability**: Idle baseline watts std should be < 10% of mean.
3. **Run consistency**: Coefficient of variation (std/mean) for J/tok should be < 15%.
4. **Token counting**: Manually verify one run's output token count against the generated text.
