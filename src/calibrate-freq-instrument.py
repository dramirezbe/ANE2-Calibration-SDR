from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from libs.instrument import KeysightHandler
from utils import ServerRealtimeConfig, ZmqPairController, ShmStore

# -----------------------------------------------------------------------------
# Calibration macros
# -----------------------------------------------------------------------------
IPC_ADDR = "ipc:///tmp/rf_engine"
INSTRUMENT_IP = "192.168.1.100"

EXPECTED_PEAK_FREQ_HZ = 100e6
CENTER_FREQ_HZ = EXPECTED_PEAK_FREQ_HZ + 1e6  # Avoid potential center spike.
HYSTERESIS_HZ = 0.1e6  # Search window around the expected peak.
SPAN_HZ = 2e6
TIMEOUT_MS = 5000

N_SHOTS = 20
SHOT_WAIT_SECONDS = 0.05
SDR_RX_TIMEOUT_SECONDS = 10.0


def _load_ppm_error(default_value: int = 0) -> int:
    """Read ppm_error from shared memory with a safe integer fallback."""
    raw_value = ShmStore().consult_persistent("ppm_error")
    if raw_value is None:
        print(f"ppm_error not found in ShmStore; using default {default_value}.")
        return int(default_value)

    try:
        return int(round(float(raw_value)))
    except (TypeError, ValueError):
        print(
            f"Invalid ppm_error value in ShmStore ({raw_value!r}); "
            f"using default {default_value}."
        )
        return int(default_value)


async def calibrate_cmd(zmq_ctrl: ZmqPairController) -> None:
    """Send one calibration command to the SDR engine."""
    print("Sending SDR calibrate command...")
    await zmq_ctrl.send_command({"calibrate": True})


def _freq_axis_hz(start_hz: float, end_hz: float, n_points: int) -> np.ndarray:
    """Build a linear frequency axis for a PSD vector."""
    if n_points <= 0:
        return np.array([], dtype=float)
    return np.linspace(float(start_hz), float(end_hz), int(n_points), dtype=float)


def _extract_sdr_trace_and_axis(
    data: dict[str, Any] | None,
    default_center_hz: float,
    default_span_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract SDR PSD vector and matching frequency axis from the ZMQ payload."""
    if not isinstance(data, dict):
        return np.array([], dtype=float), np.array([], dtype=float)

    pxx: Any = data.get("Pxx")
    start_hz = data.get("start_freq_hz")
    end_hz = data.get("end_freq_hz")

    trace = np.asarray(pxx if pxx is not None else [], dtype=float).ravel()
    if trace.size == 0:
        return trace, np.array([], dtype=float)

    if start_hz is None or end_hz is None:
        start_hz = default_center_hz - default_span_hz / 2.0
        end_hz = default_center_hz + default_span_hz / 2.0

    return trace, _freq_axis_hz(float(start_hz), float(end_hz), trace.size)


def _find_peak_with_hysteresis(
    freqs_hz: np.ndarray,
    pxx: np.ndarray,
    expected_hz: float,
    hysteresis_hz: float,
) -> tuple[float, float, bool]:
    """Find the spectral peak near the expected frequency inside a hysteresis window.

    Returns
    -------
    tuple[float, float, bool]
        Peak frequency [Hz], peak level, and whether the peak was found within
        the hysteresis window. If the window is empty, global maximum is used.
    """
    if freqs_hz.size == 0 or pxx.size == 0 or freqs_hz.size != pxx.size:
        return np.nan, np.nan, False

    in_window = np.abs(freqs_hz - expected_hz) <= hysteresis_hz
    if np.any(in_window):
        local_idx = int(np.argmax(pxx[in_window]))
        global_idx = int(np.flatnonzero(in_window)[local_idx])
        return float(freqs_hz[global_idx]), float(pxx[global_idx]), True

    fallback_idx = int(np.argmax(pxx))
    return float(freqs_hz[fallback_idx]), float(pxx[fallback_idx]), False


async def _acquire_sdr_once(
    zmq_ctrl: ZmqPairController,
    payload_sdr: dict[str, Any],
) -> dict[str, Any] | None:
    """Request one SDR realtime PSD snapshot and await response."""
    await zmq_ctrl.send_command(payload_sdr)
    return await asyncio.wait_for(zmq_ctrl.wait_for_data(), timeout=SDR_RX_TIMEOUT_SECONDS)


async def _acquire_instrument_once(
    handler: KeysightHandler,
    center_freq_hz: float,
    span_hz: float,
) -> np.ndarray:
    """Request one Keysight trace snapshot."""
    return await handler.get_trace(center_freq_hz=center_freq_hz, span_hz=span_hz)


async def main() -> None:
    """Run repeated concurrent SDR/Keysight acquisitions and report frequency error."""
    ppm_error = _load_ppm_error(default_value=0)

    config_obj = ServerRealtimeConfig(
        method_psd="welch",
        center_freq_hz=int(CENTER_FREQ_HZ),
        sample_rate_hz=int(SPAN_HZ),
        rbw_hz=int(1e3),
        window="hamming",
        overlap=0.5,
        lna_gain=8,
        vga_gain=8,
        antenna_amp=False,
        antenna_port=1,
        ppm_error=ppm_error,
        cooldown_request=0.01,
    )
    payload_sdr = asdict(config_obj)

    errors_hz: list[float] = []
    valid_shots_for_plot: list[dict[str, np.ndarray | float | int]] = []

    async with ZmqPairController(IPC_ADDR, is_server=True, verbose=True) as zmq_ctrl:
        # Enable only if SDR must be recalibrated before running the loop.
        # await calibrate_cmd(zmq_ctrl)

        async with KeysightHandler(ip=INSTRUMENT_IP, timeout_ms=TIMEOUT_MS) as handler:
            instrument_info = await handler.get_info()
            print(f"Keysight IDN: {instrument_info or 'N/A'}")
            await handler.clear_errors()

            for shot in range(1, N_SHOTS + 1):
                sdr_task = _acquire_sdr_once(zmq_ctrl, payload_sdr)
                inst_task = _acquire_instrument_once(
                    handler,
                    center_freq_hz=config_obj.center_freq_hz,
                    span_hz=config_obj.sample_rate_hz,
                )
                sdr_data, inst_trace = await asyncio.gather(sdr_task, inst_task)

                sdr_trace, sdr_freqs = _extract_sdr_trace_and_axis(
                    sdr_data,
                    default_center_hz=config_obj.center_freq_hz,
                    default_span_hz=config_obj.sample_rate_hz,
                )

                inst_trace = np.asarray(inst_trace, dtype=float).ravel()
                inst_freqs = _freq_axis_hz(
                    config_obj.center_freq_hz - config_obj.sample_rate_hz / 2.0,
                    config_obj.center_freq_hz + config_obj.sample_rate_hz / 2.0,
                    inst_trace.size,
                )

                sdr_peak_hz, sdr_peak_db, sdr_in_window = _find_peak_with_hysteresis(
                    sdr_freqs,
                    sdr_trace,
                    expected_hz=EXPECTED_PEAK_FREQ_HZ,
                    hysteresis_hz=HYSTERESIS_HZ,
                )
                inst_peak_hz, inst_peak_db, inst_in_window = _find_peak_with_hysteresis(
                    inst_freqs,
                    inst_trace,
                    expected_hz=EXPECTED_PEAK_FREQ_HZ,
                    hysteresis_hz=HYSTERESIS_HZ,
                )

                if np.isnan(sdr_peak_hz) or np.isnan(inst_peak_hz):
                    print(f"[{shot:02d}/{N_SHOTS}] Incomplete trace data; shot skipped.")
                else:
                    error_hz = sdr_peak_hz - inst_peak_hz
                    errors_hz.append(error_hz)
                    valid_shots_for_plot.append(
                        {
                            "shot": shot,
                            "sdr_freqs": sdr_freqs.copy(),
                            "sdr_trace": sdr_trace.copy(),
                            "inst_freqs": inst_freqs.copy(),
                            "inst_trace": inst_trace.copy(),
                            "sdr_peak_hz": sdr_peak_hz,
                            "inst_peak_hz": inst_peak_hz,
                        }
                    )

                    sdr_tag = "OK" if sdr_in_window else "FALLBACK"
                    inst_tag = "OK" if inst_in_window else "FALLBACK"
                    print(
                        f"[{shot:02d}/{N_SHOTS}] "
                        f"SDR peak={sdr_peak_hz/1e6:10.6f} MHz ({sdr_peak_db:7.2f} dB, {sdr_tag}) | "
                        f"Keysight peak={inst_peak_hz/1e6:10.6f} MHz ({inst_peak_db:7.2f} dB, {inst_tag}) | "
                        f"error={error_hz:+9.1f} Hz"
                    )

                await asyncio.sleep(SHOT_WAIT_SECONDS)

    if errors_hz:
        errors = np.asarray(errors_hz, dtype=float)
        print("\nFrequency error summary (SDR - Keysight):")
        print(f"  valid shots: {errors.size}/{N_SHOTS}")
        print(f"  median: {np.median(errors):+.2f} Hz")
        print(f"  mean:   {np.mean(errors):+.2f} Hz")
        print(f"  std:    {np.std(errors):.2f} Hz")
        print(f"  min:    {np.min(errors):+.2f} Hz")
        print(f"  max:    {np.max(errors):+.2f} Hz")

        # Plot one random valid realization to visually verify peak alignment.
        random_idx = int(np.random.randint(0, len(valid_shots_for_plot)))
        picked = valid_shots_for_plot[random_idx]

        sdr_freqs_mhz = np.asarray(picked["sdr_freqs"], dtype=float) / 1e6
        inst_freqs_mhz = np.asarray(picked["inst_freqs"], dtype=float) / 1e6
        sdr_trace = np.asarray(picked["sdr_trace"], dtype=float)
        inst_trace = np.asarray(picked["inst_trace"], dtype=float)

        plt.figure(figsize=(11, 5.5))
        plt.plot(sdr_freqs_mhz, sdr_trace, label="SDR Pxx", linewidth=1.4, alpha=0.9)
        plt.plot(inst_freqs_mhz, inst_trace, label="Keysight TRACE1", linewidth=1.2, alpha=0.85)
        plt.axvline(EXPECTED_PEAK_FREQ_HZ / 1e6, color="black", linestyle="--", linewidth=1.0, label="Expected peak")
        plt.axvline(float(picked["sdr_peak_hz"]) / 1e6, color="tab:blue", linestyle=":", linewidth=1.2, label="SDR peak")
        plt.axvline(float(picked["inst_peak_hz"]) / 1e6, color="tab:orange", linestyle=":", linewidth=1.2, label="Keysight peak")
        plt.title(f"Random realization #{int(picked['shot']):02d}: SDR vs Keysight")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power (dB)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

        # Plot histogram of per-shot frequency errors.
        median_error_hz = float(np.median(errors))
        plt.figure(figsize=(8.5, 4.8))
        plt.hist(errors, bins=min(12, max(5, errors.size // 2)), color="steelblue", alpha=0.85, edgecolor="black")
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="0 Hz")
        plt.axvline(median_error_hz, color="crimson", linestyle=":", linewidth=1.5, label=f"Median = {median_error_hz:+.2f} Hz")
        plt.title("Frequency Error Distribution (SDR - Keysight)")
        plt.xlabel("Error (Hz)")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        print("No valid shots were acquired; unable to compute frequency error statistics.")


if __name__ == "__main__":
    asyncio.run(main())
