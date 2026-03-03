"""Tests for src/hardware.py — trapezoidal integration and SLURM GPU mapping."""

import os
from unittest.mock import patch

from src.hardware import PowerSampler, _resolve_nvml_index


class TestTrapezoidalIntegration:
    """Test PowerSampler.get_results() energy calculation with known power curves."""

    def _sampler_with_samples(self, samples):
        """Create a PowerSampler and inject known samples."""
        ps = PowerSampler(baseline_watts=0.0)
        ps._samples = samples
        ps._running = False
        return ps

    def test_constant_power(self):
        """100W for 10 seconds = 1000 J."""
        samples = [(float(t), 100.0) for t in range(11)]  # 0..10s
        ps = self._sampler_with_samples(samples)
        result = ps.get_results()
        assert abs(result.total_energy_joules - 1000.0) < 1e-6
        assert result.duration_seconds == 10.0

    def test_linear_ramp(self):
        """Power ramps from 0W to 200W over 10s. Trapezoidal = 1000 J."""
        samples = [(float(t), 20.0 * t) for t in range(11)]
        ps = self._sampler_with_samples(samples)
        result = ps.get_results()
        # Integral of linear 0->200 over 10s = (0+200)/2 * 10 = 1000
        assert abs(result.total_energy_joules - 1000.0) < 1e-6

    def test_baseline_subtraction(self):
        """Net energy subtracts baseline power over duration."""
        samples = [(float(t), 150.0) for t in range(11)]
        ps = PowerSampler(baseline_watts=50.0)
        ps._samples = samples
        ps._running = False
        result = ps.get_results()
        # Total = 150 * 10 = 1500 J
        # Net = 1500 - 50 * 10 = 1000 J
        assert abs(result.total_energy_joules - 1500.0) < 1e-6
        assert abs(result.net_energy_joules - 1000.0) < 1e-6

    def test_empty_samples(self):
        """No samples should return zeros."""
        ps = self._sampler_with_samples([])
        result = ps.get_results()
        assert result.total_energy_joules == 0.0
        assert result.sample_count == 0

    def test_single_sample(self):
        """One sample cannot integrate — returns zero energy."""
        ps = self._sampler_with_samples([(0.0, 100.0)])
        result = ps.get_results()
        assert result.total_energy_joules == 0.0
        assert result.sample_count == 1

    def test_peak_and_mean(self):
        """Verify peak and mean watts from known data."""
        samples = [(0.0, 100.0), (1.0, 200.0), (2.0, 150.0)]
        ps = self._sampler_with_samples(samples)
        result = ps.get_results()
        assert result.peak_watts == 200.0
        assert abs(result.mean_inference_watts - 150.0) < 1e-6

    def test_irregular_intervals(self):
        """Trapezoidal works with non-uniform time steps."""
        samples = [
            (0.0, 100.0),
            (0.1, 100.0),
            (0.5, 100.0),
            (2.0, 100.0),
        ]
        ps = self._sampler_with_samples(samples)
        result = ps.get_results()
        # Constant 100W for 2s = 200 J
        assert abs(result.total_energy_joules - 200.0) < 1e-6

    def test_step_function(self):
        """Power jumps from 100W to 300W at t=5."""
        samples = [(float(t), 100.0) for t in range(6)]
        samples += [(float(t), 300.0) for t in range(6, 11)]
        ps = self._sampler_with_samples(samples)
        result = ps.get_results()
        # 0-5: 100W*5 = 500 J
        # 5-6: trapezoid (100+300)/2 * 1 = 200 J
        # 6-10: 300W*4 = 1200 J
        # Total = 500 + 200 + 1200 = 1900 J
        assert abs(result.total_energy_joules - 1900.0) < 1e-6


class TestResolveNvmlIndex:
    """Test SLURM CUDA_VISIBLE_DEVICES mapping."""

    def test_no_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            assert _resolve_nvml_index(0) == 0

    def test_integer_mapping(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            assert _resolve_nvml_index(0) == 3

    def test_multi_gpu_mapping(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
            assert _resolve_nvml_index(0) == 2
            assert _resolve_nvml_index(1) == 5

    def test_out_of_range(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            assert _resolve_nvml_index(5) == 5

    def test_trailing_comma(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,"}):
            assert _resolve_nvml_index(0) == 2
