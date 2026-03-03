"""Layer 1: Hardware Instrumentation — GPU power sampling via NVML.

Samples GPU power draw at high frequency during inference and computes
total energy consumption via trapezoidal integration. Supports idle
baseline measurement and baseline subtraction for net inference energy.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

import pynvml

logger = logging.getLogger(__name__)


@dataclass
class EnergyMeasurement:
    """Results from a power sampling session."""

    total_energy_joules: float
    net_energy_joules: float  # after baseline subtraction
    baseline_watts: float
    mean_inference_watts: float
    peak_watts: float
    duration_seconds: float
    sample_count: int
    samples: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, watts)


def _resolve_nvml_index(cuda_index: int = 0) -> int:
    """Map CUDA device index to physical NVML index under SLURM.

    When SLURM sets CUDA_VISIBLE_DEVICES (e.g. "3" or a GPU UUID),
    CUDA sees it as device 0 but NVML still uses the physical index.
    This resolves the physical index for correct NVML power queries.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return cuda_index

    entries = [e.strip() for e in cvd.split(",") if e.strip()]
    if cuda_index >= len(entries):
        return cuda_index

    entry = entries[cuda_index]

    # Integer index — direct physical mapping
    try:
        return int(entry)
    except ValueError:
        pass

    # UUID/MIG format — resolve via NVML
    try:
        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                dev_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(dev_uuid, bytes):
                    dev_uuid = dev_uuid.decode()
                if entry in dev_uuid or dev_uuid in entry:
                    return i
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass

    return cuda_index


def measure_idle_baseline(gpu_index: int = 0, duration_seconds: float = 30.0,
                          interval_seconds: float = 0.1) -> float:
    """Measure mean idle GPU power draw over a given duration.

    Samples power at the given interval, computes mean and checks stability
    (std < 10% of mean). If unstable, extends sampling to 60 seconds.

    Args:
        gpu_index: CUDA device index (will be resolved to physical NVML index).
        duration_seconds: How long to sample idle power.
        interval_seconds: Sampling interval in seconds.

    Returns:
        Mean idle power in watts.
    """
    nvml_index = _resolve_nvml_index(gpu_index)

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)

        def _sample(duration: float) -> list[float]:
            samples = []
            end_time = time.monotonic() + duration
            while time.monotonic() < end_time:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    samples.append(power_mw / 1000.0)
                except pynvml.NVMLError:
                    pass
                time.sleep(interval_seconds)
            return samples

        # First pass: 30 seconds
        logger.info("Measuring idle baseline for %.0fs...", duration_seconds)
        samples = _sample(duration_seconds)

        if not samples:
            raise RuntimeError("No power samples collected during idle baseline")

        mean_watts = sum(samples) / len(samples)
        variance = sum((s - mean_watts) ** 2 for s in samples) / (len(samples) - 1) if len(samples) > 1 else 0.0
        std_watts = variance ** 0.5

        # Check stability
        if mean_watts > 0 and std_watts / mean_watts > 0.10:
            logger.warning(
                "Baseline unstable (std=%.1fW, mean=%.1fW, CV=%.1f%%). "
                "Extending to 60s...",
                std_watts, mean_watts, 100 * std_watts / mean_watts,
            )
            extended_samples = _sample(60.0)
            if extended_samples:
                samples = extended_samples
                mean_watts = sum(samples) / len(samples)
                variance = sum((s - mean_watts) ** 2 for s in samples) / (len(samples) - 1)
                std_watts = variance ** 0.5

        logger.info(
            "Idle baseline: %.1fW (std=%.1fW, %d samples)",
            mean_watts, std_watts, len(samples),
        )
        return mean_watts
    finally:
        pynvml.nvmlShutdown()


class PowerSampler:
    """Context manager that samples GPU power in a background thread.

    Usage:
        baseline = measure_idle_baseline()
        with PowerSampler(baseline_watts=baseline) as sampler:
            # ... run inference ...
        result = sampler.get_results()
    """

    def __init__(self, gpu_index: int = 0, interval_seconds: float = 0.1,
                 baseline_watts: float = 0.0):
        self._gpu_index = gpu_index
        self._interval = interval_seconds
        self._baseline = baseline_watts
        self._samples: list[tuple[float, float]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._handle = None

    def __enter__(self) -> PowerSampler:
        nvml_index = _resolve_nvml_index(self._gpu_index)
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        """Background sampling loop — appends (timestamp, watts) tuples."""
        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                with self._lock:
                    self._samples.append((time.monotonic(), power_mw / 1000.0))
            except pynvml.NVMLError:
                pass
            time.sleep(self._interval)

    def __exit__(self, *exc) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def get_results(self) -> EnergyMeasurement:
        """Compute energy from collected power samples using trapezoidal integration."""
        with self._lock:
            samples = list(self._samples)
        if len(samples) < 2:
            return EnergyMeasurement(
                total_energy_joules=0.0,
                net_energy_joules=0.0,
                baseline_watts=self._baseline,
                mean_inference_watts=0.0,
                peak_watts=0.0,
                duration_seconds=0.0,
                sample_count=len(samples),
                samples=samples,
            )

        # Trapezoidal integration
        total_energy = sum(
            (samples[i][1] + samples[i - 1][1]) / 2.0
            * (samples[i][0] - samples[i - 1][0])
            for i in range(1, len(samples))
        )

        duration = samples[-1][0] - samples[0][0]
        net_energy = total_energy - (self._baseline * duration)
        mean_watts = sum(s[1] for s in samples) / len(samples)
        peak_watts = max(s[1] for s in samples)

        return EnergyMeasurement(
            total_energy_joules=total_energy,
            net_energy_joules=net_energy,
            baseline_watts=self._baseline,
            mean_inference_watts=mean_watts,
            peak_watts=peak_watts,
            duration_seconds=duration,
            sample_count=len(samples),
            samples=samples,
        )
