"""Tests for src/metrics.py — compute_metrics and aggregate_runs."""

import math

from src.hardware import EnergyMeasurement
from src.inference import InferenceResult
from src.metrics import compute_metrics, aggregate_runs, BenchmarkMetrics


def _make_energy(total_j=10.0, net_j=8.0, baseline=50.0, mean_w=150.0,
                 peak_w=200.0, duration=2.0, n_samples=20):
    return EnergyMeasurement(
        total_energy_joules=total_j,
        net_energy_joules=net_j,
        baseline_watts=baseline,
        mean_inference_watts=mean_w,
        peak_watts=peak_w,
        duration_seconds=duration,
        sample_count=n_samples,
    )


def _make_inference(prompt_tokens=50, output_tokens=200, gen_time=2.0,
                    batch_size=1):
    return InferenceResult(
        prompt_id="test",
        task_type="qa",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        generation_time_seconds=gen_time,
        output_text="test output",
        batch_size=batch_size,
    )


class TestComputeMetrics:
    def test_basic_j_per_tok(self):
        energy = _make_energy(net_j=10.0)
        inference = _make_inference(output_tokens=100)
        m = compute_metrics(energy, inference)
        assert m.joules_per_token == 0.1  # 10 / 100

    def test_tokens_per_second(self):
        energy = _make_energy()
        inference = _make_inference(output_tokens=200, gen_time=4.0)
        m = compute_metrics(energy, inference)
        assert m.tokens_per_second == 50.0  # 200 / 4.0

    def test_carbon_calculation(self):
        energy = _make_energy(net_j=3600.0)  # 1 Wh
        inference = _make_inference(output_tokens=1)
        m = compute_metrics(energy, inference, grid_carbon_intensity=230.0)
        # J/tok = 3600, kWh/tok = 3600/3600000 = 0.001
        # gCO2eq/tok = 0.001 * 230 = 0.23
        assert abs(m.gco2eq_per_token - 0.23) < 1e-6

    def test_no_carbon_without_intensity(self):
        energy = _make_energy()
        inference = _make_inference()
        m = compute_metrics(energy, inference)
        assert m.gco2eq_per_token is None

    def test_watts_and_peak(self):
        energy = _make_energy(mean_w=175.0, peak_w=250.0)
        inference = _make_inference()
        m = compute_metrics(energy, inference)
        assert m.mean_watts == 175.0
        assert m.peak_watts == 250.0


class TestAggregateRuns:
    def _make_runs(self, j_per_tok_values):
        runs = []
        for j in j_per_tok_values:
            runs.append(BenchmarkMetrics(
                joules_per_token=j,
                tokens_per_second=100.0,
                mean_watts=150.0,
                peak_watts=200.0,
                total_energy_joules=10.0,
                net_energy_joules=j * 200,
                output_tokens=200,
                input_tokens=50,
                generation_time_seconds=2.0,
            ))
        return runs

    def test_mean_calculation(self):
        runs = self._make_runs([0.1, 0.1, 0.1, 0.1, 0.1])
        agg = aggregate_runs(runs, "test", "qa", 1)
        assert abs(agg.joules_per_token.mean - 0.1) < 1e-9

    def test_std_calculation(self):
        # Known values: [2, 4, 4, 4, 5, 5, 7, 9] -> std (sample) = 2.138
        runs = self._make_runs([2, 4, 4, 4, 5, 5, 7, 9])
        agg = aggregate_runs(runs, "test", "qa", 1)
        expected_mean = 5.0
        expected_var = sum((v - expected_mean) ** 2 for v in [2, 4, 4, 4, 5, 5, 7, 9]) / 7
        expected_std = math.sqrt(expected_var)
        assert abs(agg.joules_per_token.mean - expected_mean) < 1e-9
        assert abs(agg.joules_per_token.std - expected_std) < 1e-6

    def test_ci_95(self):
        values = [0.10, 0.12, 0.11, 0.09, 0.10]
        runs = self._make_runs(values)
        agg = aggregate_runs(runs, "test", "qa", 1)
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        margin = 1.96 * std / math.sqrt(n)
        assert abs(agg.joules_per_token.ci_95_lower - (mean - margin)) < 1e-9
        assert abs(agg.joules_per_token.ci_95_upper - (mean + margin)) < 1e-9

    def test_single_run(self):
        runs = self._make_runs([0.05])
        agg = aggregate_runs(runs, "test", "qa", 1)
        assert agg.joules_per_token.mean == 0.05
        assert agg.joules_per_token.std == 0.0
        assert agg.n_runs == 1

    def test_metadata(self):
        runs = self._make_runs([0.1, 0.2])
        agg = aggregate_runs(runs, "sum_short", "summarisation", 4)
        assert agg.prompt_id == "sum_short"
        assert agg.task_type == "summarisation"
        assert agg.batch_size == 4
        assert agg.n_runs == 2
