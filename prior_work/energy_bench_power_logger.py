"""NVML-based GPU power logger for energy measurement.

Samples GPU power draw in a background thread and computes
total energy (joules) via trapezoidal integration.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List

import pynvml


def resolve_nvml_gpu_index(cuda_index: int) -> int:
    """Map a CUDA device index to the physical NVML GPU index.

    When CUDA_VISIBLE_DEVICES is set (e.g. by SLURM), CUDA device 0
    maps to a different physical GPU than NVML device 0.  This function
    translates the CUDA-relative index back to the physical one that
    NVML expects.

    Handles integer indices ("2,3"), GPU UUIDs ("GPU-xxxx-..."),
    MIG identifiers ("MIG-GPU-xxxx/.../..."), and trailing commas.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return cuda_index

    # Filter empty entries (e.g. trailing comma: "0,1,")
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
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
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


@dataclass
class PowerSample:
    timestamp: float  # time.monotonic()
    power_w: float
    gpu_clock_mhz: int = 0  # GPU SM clock at sample time


@dataclass
class PhaseMarker:
    """Marks the boundary between inference phases (prefill vs decode)."""
    phase_name: str  # "prefill_start", "prefill_end", "decode_start", "decode_end"
    timestamp: float
    sample_index: int  # Index in samples list when this phase started


class PowerLogger:
    """Logs GPU power draw in a background thread using NVML."""

    def __init__(self, gpu_index: int = 0, sample_interval_s: float = 0.1):
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_s
        self.samples: List[PowerSample] = []
        self.phase_markers: List[PhaseMarker] = []
        self._stop_event = threading.Event()
        self._thread = None
        self._handle = None

    def start(self):
        """Begin sampling GPU power in a background thread."""
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.samples = []
        self.phase_markers = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[PowerSample]:
        """Stop sampling and return collected samples."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
        return self.samples

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_w = power_mw / 1000.0
                try:
                    clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_SM
                    )
                except pynvml.NVMLError:
                    clock_mhz = 0
                self.samples.append(PowerSample(
                    timestamp=time.monotonic(),
                    power_w=power_w,
                    gpu_clock_mhz=clock_mhz,
                ))
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self.sample_interval_s)

    def get_energy_joules(self) -> float:
        """Compute total energy via trapezoidal integration of power samples."""
        if len(self.samples) < 2:
            return 0.0
        joules = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i].timestamp - self.samples[i - 1].timestamp
            avg_power = (self.samples[i].power_w + self.samples[i - 1].power_w) / 2.0
            joules += avg_power * dt
        return joules

    def get_avg_power(self) -> float:
        """Return mean power draw across all samples."""
        if not self.samples:
            return 0.0
        return sum(s.power_w for s in self.samples) / len(self.samples)

    def get_duration(self) -> float:
        """Return wall-clock duration covered by the samples."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp

    def mark_phase(self, phase_name: str):
        """Mark a phase transition during inference.
        
        Args:
            phase_name: Name of phase (e.g., "prefill_start", "decode_start")
        
        Example:
            logger.start()
            logger.mark_phase("prefill_start")
            # ... prompt processing ...
            logger.mark_phase("prefill_end")
            logger.mark_phase("decode_start")
            # ... token generation ...
            logger.mark_phase("decode_end")
            logger.stop()
        """
        marker = PhaseMarker(
            phase_name=phase_name,
            timestamp=time.monotonic(),
            sample_index=len(self.samples)
        )
        self.phase_markers.append(marker)

    def get_phase_energy(self, start_phase: str, end_phase: str) -> float:
        """Calculate energy consumed during a specific phase.

        Uses timestamp-based interpolation for accuracy even when
        few samples fall within the phase boundaries.

        Args:
            start_phase: Phase marker name for start (e.g., "prefill_start")
            end_phase: Phase marker name for end (e.g., "prefill_end")

        Returns:
            Energy in joules for that phase
        """
        start_marker = None
        end_marker = None

        for marker in self.phase_markers:
            if marker.phase_name == start_phase:
                start_marker = marker
            if marker.phase_name == end_phase:
                end_marker = marker

        if not start_marker or not end_marker:
            return 0.0

        phase_start_t = start_marker.timestamp
        phase_end_t = end_marker.timestamp
        phase_duration = phase_end_t - phase_start_t

        if phase_duration <= 0 or len(self.samples) < 2:
            return 0.0

        # Find samples that overlap with this phase window
        # Use all samples whose time range intersects [phase_start_t, phase_end_t]
        joules = 0.0
        for i in range(1, len(self.samples)):
            s_prev = self.samples[i - 1]
            s_curr = self.samples[i]

            # Clip this sample interval to the phase window
            interval_start = max(s_prev.timestamp, phase_start_t)
            interval_end = min(s_curr.timestamp, phase_end_t)

            if interval_end <= interval_start:
                continue

            dt = interval_end - interval_start
            avg_power = (s_prev.power_w + s_curr.power_w) / 2.0
            joules += avg_power * dt

        # Fallback: if no sample intervals overlapped (phase shorter than
        # any sample gap), estimate from nearest samples and duration
        if joules == 0.0 and self.samples:
            nearest_power = self._interpolate_power_at(phase_start_t)
            joules = nearest_power * phase_duration

        return joules

    def _interpolate_power_at(self, timestamp: float) -> float:
        """Interpolate power at a given timestamp from surrounding samples."""
        if not self.samples:
            return 0.0
        if len(self.samples) == 1:
            return self.samples[0].power_w

        # Find surrounding samples
        for i in range(1, len(self.samples)):
            if self.samples[i].timestamp >= timestamp:
                s0 = self.samples[i - 1]
                s1 = self.samples[i]
                if s1.timestamp == s0.timestamp:
                    return s0.power_w
                frac = (timestamp - s0.timestamp) / (s1.timestamp - s0.timestamp)
                return s0.power_w + frac * (s1.power_w - s0.power_w)

        return self.samples[-1].power_w

    @staticmethod
    def get_gpu_info(gpu_index: int = 0) -> dict:
        """Query static GPU metadata via NVML."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        info = {
            "gpu_name": name.decode() if isinstance(name, bytes) else name,
            "driver_version": driver.decode() if isinstance(driver, bytes) else driver,
        }
        try:
            info["gpu_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_GRAPHICS
            )
            info["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            )
        except pynvml.NVMLError:
            pass
        try:
            info["power_limit_w"] = (
                pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            )
        except pynvml.NVMLError:
            pass
        pynvml.nvmlShutdown()
        return info
