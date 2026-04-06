"""Example script showing end-to-end usage of ``libs.instrument.KeysightHandler``.

The script demonstrates all public methods of :class:`KeysightHandler`:

1. Open/close the VISA session through ``async with``.
2. Query instrument identity via :meth:`get_info`.
3. Clear status/error registers via :meth:`clear_errors`.
4. Acquire ``TRACE1`` data via :meth:`get_trace`.
5. Plot the resulting spectrum trace with Matplotlib.

Run
---
python src/keysight-usage.py
"""

from __future__ import annotations

import asyncio

import matplotlib.pyplot as plt
import numpy as np

from libs.instrument import KeysightHandler

# -----------------------------------------------------------------------------
# Example configuration macros
# -----------------------------------------------------------------------------
INSTRUMENT_IP = "192.168.1.100"
CENTER_FREQ_HZ = 100e6
SPAN_HZ = 20e6
TIMEOUT_MS = 5000


def _plot_trace(trace: np.ndarray, center_freq_hz: float, span_hz: float) -> None:
	"""Plot the acquired trace against a computed frequency axis."""
	if trace.size == 0:
		print("No trace data received; skipping plot.")
		return

	f_start = center_freq_hz - span_hz / 2.0
	f_stop = center_freq_hz + span_hz / 2.0
	freq_axis_hz = np.linspace(f_start, f_stop, trace.size)

	plt.figure(figsize=(10, 5))
	plt.plot(freq_axis_hz / 1e6, trace, linewidth=1.4, color="steelblue")
	plt.title("Keysight TRACE1 Spectrum")
	plt.xlabel("Frequency (MHz)")
	plt.ylabel("Amplitude (dB)")
	plt.grid(True, linestyle="--", alpha=0.3)
	plt.tight_layout()
	plt.show()


async def _run_demo() -> None:
	"""Connect to instrument, call all handler methods, and render the trace."""
	async with KeysightHandler(ip=INSTRUMENT_IP, timeout_ms=TIMEOUT_MS) as handler:
		instrument_info = await handler.get_info()
		print(f"Instrument IDN: {instrument_info or 'N/A'}")

		await handler.clear_errors()

		trace = await handler.get_trace(
			center_freq_hz=CENTER_FREQ_HZ,
			span_hz=SPAN_HZ,
		)
		print(f"Trace samples: {trace.size}")

	_plot_trace(
		trace=trace,
		center_freq_hz=CENTER_FREQ_HZ,
		span_hz=SPAN_HZ,
	)


def main() -> None:
	"""Run the Keysight usage example using module-level configuration macros."""
	asyncio.run(_run_demo())


if __name__ == "__main__":
	main()
