"""Layer 3: Metric Computation — combine hardware and inference measurements.

Computes all benchmark metrics from raw power samples and inference results,
including statistical aggregation across multiple runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .hardware import EnergyMeasurement
from .inference import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics for a single benchmark run."""

    joules_per_token: float
    tokens_per_second: float
    mean_watts: float
    peak_watts: float
    total_energy_joules: float
    net_energy_joules: float
    output_tokens: int
    input_tokens: int
    generation_time_seconds: float
    gco2eq_per_token: float | None = None


@dataclass
class AggregatedMetrics:
    """Statistical aggregation of metrics across multiple runs."""

    prompt_id: str
    task_type: str
    batch_size: int
    n_runs: int
    joules_per_token: StatSummary
    tokens_per_second: StatSummary
    mean_watts: StatSummary
    output_tokens: StatSummary
    input_tokens: StatSummary


@dataclass
class StatSummary:
    """Mean, std, and 95% confidence interval for a metric."""

    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float


def compute_metrics(energy: EnergyMeasurement, inference: InferenceResult,
                    grid_carbon_intensity: float | None = None) -> BenchmarkMetrics:
    """Compute all metrics from a single run's measurements.

    Args:
        energy: Power sampling results.
        inference: Inference execution results.
        grid_carbon_intensity: Grid carbon intensity in gCO2/kWh (optional).

    Returns:
        BenchmarkMetrics for this run.
    """
    if inference.output_tokens <= 0:
        raise ValueError(
            f"Cannot compute metrics: output_tokens={inference.output_tokens} "
            f"(prompt_id={inference.prompt_id}, batch_size={inference.batch_size})"
        )
    if inference.generation_time_seconds <= 0:
        raise ValueError(
            f"Cannot compute metrics: generation_time={inference.generation_time_seconds}s"
        )
    joules_per_token = energy.net_energy_joules / inference.output_tokens
    tokens_per_second = inference.output_tokens / inference.generation_time_seconds

    gco2eq_per_token = None
    if grid_carbon_intensity is not None:
        kwh_per_token = joules_per_token / 3_600_000
        gco2eq_per_token = kwh_per_token * grid_carbon_intensity

    return BenchmarkMetrics(
        joules_per_token=joules_per_token,
        tokens_per_second=tokens_per_second,
        mean_watts=energy.mean_inference_watts,
        peak_watts=energy.peak_watts,
        total_energy_joules=energy.total_energy_joules,
        net_energy_joules=energy.net_energy_joules,
        output_tokens=inference.output_tokens,
        input_tokens=inference.prompt_tokens,
        generation_time_seconds=inference.generation_time_seconds,
        gco2eq_per_token=gco2eq_per_token,
    )


def aggregate_runs(runs: list[BenchmarkMetrics], prompt_id: str = "",
                   task_type: str = "", batch_size: int = 1) -> AggregatedMetrics:
    """Compute mean, std, and 95% CI across multiple runs of the same configuration.

    Args:
        runs: List of BenchmarkMetrics from repeated runs.
        prompt_id: Identifier for the prompt.
        task_type: Category of the prompt.
        batch_size: Batch size used.

    Returns:
        AggregatedMetrics with statistical summaries.
    """
    def _summarise(values: list[float]) -> StatSummary:
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(variance)
        ci_margin = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        return StatSummary(
            mean=mean,
            std=std,
            ci_95_lower=mean - ci_margin,
            ci_95_upper=mean + ci_margin,
        )

    return AggregatedMetrics(
        prompt_id=prompt_id,
        task_type=task_type,
        batch_size=batch_size,
        n_runs=len(runs),
        joules_per_token=_summarise([r.joules_per_token for r in runs]),
        tokens_per_second=_summarise([r.tokens_per_second for r in runs]),
        mean_watts=_summarise([r.mean_watts for r in runs]),
        output_tokens=_summarise([float(r.output_tokens) for r in runs]),
        input_tokens=_summarise([float(r.input_tokens) for r in runs]),
    )
