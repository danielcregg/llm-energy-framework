"""Layer 4: Main Orchestrator — run full benchmarks and produce structured reports.

Orchestrates the end-to-end benchmark pipeline: load model, measure idle baseline,
run all prompts at all batch sizes for N runs each, aggregate statistics, and
output structured JSON reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from .hardware import measure_idle_baseline, PowerSampler
from .inference import load_model, load_prompts, run_inference
from .metrics import compute_metrics, aggregate_runs

logger = logging.getLogger(__name__)

FRAMEWORK_VERSION = "0.1.0"
WARMUP_RUNS = 3
DEFAULT_N_RUNS = 10
DEFAULT_MAX_NEW_TOKENS = 200


def _get_hardware_info() -> dict:
    """Collect hardware environment information."""
    # TODO: Populate with nvidia-smi and system info on GPU machine
    info = {
        "gpu_name": "unknown",
        "gpu_memory_gb": 0,
        "cuda_version": torch.version.cuda or "unknown",
        "driver_version": "unknown",
        "cpu_model": platform.processor() or "unknown",
        "ram_gb": 0,
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info["gpu_memory_gb"] = round(mem.total / (1024 ** 3), 1)
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
    except Exception:
        logger.warning("Could not query NVML for GPU info")
    return info


def _get_software_info() -> dict:
    """Collect software environment information."""
    import transformers
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "os": f"{platform.system()} {platform.release()}",
    }


def run_benchmark(model_name: str, precision: str = "fp16",
                  batch_sizes: list[int] | None = None,
                  n_runs: int = DEFAULT_N_RUNS,
                  max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                  output_dir: str = "results",
                  grid_carbon_intensity: float | None = None) -> dict:
    """Run a full benchmark suite for one model.

    Args:
        model_name: Hugging Face model identifier.
        precision: Precision level ('fp16', 'int8', 'int4').
        batch_sizes: List of batch sizes to test.
        n_runs: Number of measured runs per configuration.
        max_new_tokens: Maximum tokens to generate per prompt.
        output_dir: Directory for output files.
        grid_carbon_intensity: Optional grid carbon intensity in gCO2/kWh.

    Returns:
        The complete benchmark report as a dict.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model: %s (precision: %s)", model_name, precision)
    model, tokenizer = load_model(model_name, precision)

    # Measure idle baseline
    logger.info("Measuring idle baseline (30s)...")
    baseline_watts = measure_idle_baseline()

    # Load prompts
    prompts = load_prompts()

    # Run benchmarks
    all_results = []
    for prompt_info in prompts:
        for batch_size in batch_sizes:
            config_label = f"{prompt_info['id']}/bs={batch_size}"

            # Warmup
            logger.info("Warmup (%d runs): %s", WARMUP_RUNS, config_label)
            for _ in range(WARMUP_RUNS):
                run_inference(model, tokenizer, prompt_info["prompt"],
                              max_new_tokens=max_new_tokens, batch_size=batch_size)

            # Measured runs
            logger.info("Measuring (%d runs): %s", n_runs, config_label)
            run_metrics = []
            for i in range(n_runs):
                with PowerSampler(baseline_watts=baseline_watts) as sampler:
                    inf_result = run_inference(
                        model, tokenizer, prompt_info["prompt"],
                        prompt_id=prompt_info["id"],
                        task_type=prompt_info["task"],
                        max_new_tokens=max_new_tokens,
                        batch_size=batch_size,
                    )
                energy = sampler.get_results()
                metrics = compute_metrics(energy, inf_result, grid_carbon_intensity)
                run_metrics.append(metrics)

            # Aggregate
            agg = aggregate_runs(run_metrics, prompt_info["id"],
                                 prompt_info["task"], batch_size)
            all_results.append(agg)

    # Build report
    report = _build_report(model_name, precision, batch_sizes, n_runs,
                           max_new_tokens, baseline_watts, all_results)

    # Save
    report_file = output_path / f"benchmark_{model_name.replace('/', '_')}_{precision}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved: %s", report_file)

    # Print summary
    _print_summary(report)

    return report


def _build_report(model_name, precision, batch_sizes, n_runs,
                  max_new_tokens, baseline_watts, results) -> dict:
    """Assemble the structured JSON report."""
    return {
        "report_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework_version": FRAMEWORK_VERSION,
        "model": {
            "name": model_name,
            "precision": precision,
        },
        "hardware": _get_hardware_info(),
        "software": _get_software_info(),
        "benchmark_config": {
            "n_runs_per_config": n_runs,
            "warmup_runs": WARMUP_RUNS,
            "max_new_tokens": max_new_tokens,
            "batch_sizes_tested": batch_sizes,
            "sampling_interval_ms": 100,
            "idle_baseline_duration_seconds": 30,
        },
        "results": [
            {
                "prompt_id": r.prompt_id,
                "task_type": r.task_type,
                "batch_size": r.batch_size,
                "runs": r.n_runs,
                "metrics": {
                    "joules_per_token": _stat_dict(r.joules_per_token),
                    "tokens_per_second": _stat_dict(r.tokens_per_second),
                    "mean_watts": _stat_dict(r.mean_watts),
                    "output_tokens": _stat_dict(r.output_tokens),
                    "input_tokens": _stat_dict(r.input_tokens),
                },
            }
            for r in results
        ],
        "summary": {
            "idle_baseline_watts": baseline_watts,
        },
    }


def _stat_dict(s) -> dict:
    return {
        "mean": round(s.mean, 4),
        "std": round(s.std, 4),
        "ci_95_lower": round(s.ci_95_lower, 4),
        "ci_95_upper": round(s.ci_95_upper, 4),
    }


def _print_summary(report: dict) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK REPORT: {report['model']['name']}")
    print(f"Precision: {report['model']['precision']}")
    print(f"GPU: {report['hardware']['gpu_name']}")
    print("=" * 70)
    print(f"{'Prompt':<15} {'Batch':>5} {'J/tok':>8} {'tok/s':>8} {'Watts':>8}")
    print("-" * 70)
    for r in report["results"]:
        m = r["metrics"]
        print(f"{r['prompt_id']:<15} {r['batch_size']:>5} "
              f"{m['joules_per_token']['mean']:>8.4f} "
              f"{m['tokens_per_second']['mean']:>8.1f} "
              f"{m['mean_watts']['mean']:>8.1f}")
    print("=" * 70)
    print(f"Idle baseline: {report['summary']['idle_baseline_watts']:.1f} W")
    print()


def main():
    parser = argparse.ArgumentParser(description="LLM Energy Benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8", "int4"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--grid-carbon-intensity", type=float, default=None,
                        help="Grid carbon intensity in gCO2/kWh")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_benchmark(
        model_name=args.model,
        precision=args.precision,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        grid_carbon_intensity=args.grid_carbon_intensity,
    )


if __name__ == "__main__":
    main()
