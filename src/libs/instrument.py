import asyncio
from typing import Optional

import numpy as np
import pyvisa

class KeysightHandler:
    """Asynchronous wrapper around a Keysight VISA TCP/IP instrument session.

    The handler manages instrument lifecycle through an async context manager,
    executes blocking VISA calls in worker threads, and exposes convenience
    methods for common spectrum-analyser operations.

    Parameters
    ----------
    ip : str
        Instrument IPv4/hostname reachable through VISA TCP/IP.
    timeout_ms : int, optional
        VISA timeout in milliseconds for read/query operations.
        Defaults to ``5000``.

    Attributes
    ----------
    ip : str
        Instrument address used to build the VISA resource string.
    timeout_ms : int
        Session timeout applied after connecting.
    rm : pyvisa.ResourceManager or None
        Active VISA resource manager.
    inst : pyvisa.resources.Resource or None
        Open instrument resource.
    """

    def __init__(self, ip: str, timeout_ms: int = 5000):
        self.ip = ip
        self.timeout_ms = timeout_ms
        self.rm: Optional[pyvisa.ResourceManager] = None
        self.inst = None

    async def __aenter__(self):
        """Open the VISA session and configure ASCII trace format.

        Returns
        -------
        KeysightHandler
            The connected handler instance.
        """
        self.rm = pyvisa.ResourceManager('@py')
        assert self.rm is not None
        self.inst = await asyncio.to_thread(self.rm.open_resource, f'TCPIP::{self.ip}::INSTR')
        self.inst.timeout = self.timeout_ms
        await self._write(':FORM ASC')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close instrument and VISA manager resources.

        Parameters
        ----------
        exc_type : type or None
            Exception type raised inside the context, if any.
        exc_val : BaseException or None
            Exception instance raised inside the context, if any.
        exc_tb : traceback or None
            Traceback associated with ``exc_val``, if any.
        """
        if self.inst:
            await asyncio.to_thread(self.inst.close)
            self.inst = None
        if self.rm:
            await asyncio.to_thread(self.rm.close)
            self.rm = None

    def _require_inst(self):
        """Return the active instrument resource or raise when disconnected."""
        if self.inst is None:
            raise RuntimeError('Instrument session is not open.')
        return self.inst

    async def _write(self, cmd: str):
        """Send a SCPI command to the instrument in a worker thread.

        Parameters
        ----------
        cmd : str
            SCPI command string.
        """
        inst = self._require_inst()
        await asyncio.to_thread(inst.write, cmd)

    async def _query(self, cmd: str):
        """Execute a SCPI query and return its string response.

        Parameters
        ----------
        cmd : str
            SCPI query command.

        Returns
        -------
        str
            Raw string returned by the instrument backend.
        """
        inst = self._require_inst()
        return await asyncio.to_thread(inst.query, cmd)

    async def _query_ascii(self, cmd: str):
        """Execute an ASCII-values query and return numeric samples.

        Parameters
        ----------
        cmd : str
            SCPI query command for ASCII numerical data.

        Returns
        -------
        list[float]
            Parsed numeric sequence as produced by PyVISA.
        """
        inst = self._require_inst()
        return await asyncio.to_thread(inst.query_ascii_values, cmd)

    async def get_info(self) -> str:
        """Read the instrument identification string.

        Returns
        -------
        str
            Trimmed response to ``*IDN?``. Returns an empty string on failure.
        """
        try:
            info = await self._query('*IDN?')
            return info.strip()
        except Exception as e:
            print(f"Error retrieving instrument information: {e}")
            return ""

    async def clear_errors(self):
        """Clear instrument status and error queue.

        Sends the SCPI ``*CLS`` command. Errors are reported to stdout and are
        otherwise ignored to preserve non-raising behaviour.
        """
        try:
            await self._write('*CLS')
        except Exception as e:
            print(f"Error clearing instrument status: {e}")

    async def get_trace(self, center_freq_hz: float, span_hz: float) -> np.ndarray:
        """Acquire TRACE1 power values for a given center frequency and span.

        Parameters
        ----------
        center_freq_hz : float
            Frequency centre in Hz.
        span_hz : float
            Frequency span in Hz.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of trace values. Returns an empty array on
            any communication or parsing failure.
        """
        try:
            await self._write(f':SENS:FREQ:CENT {center_freq_hz}')
            await self._write(f':SENS:FREQ:SPAN {span_hz}')

            trace_data = await self._query_ascii(':TRACe:DATA? TRACE1')
            return np.asarray(trace_data, dtype=float)
        except Exception as e:
            print(f"Error retrieving trace data: {e}")
            return np.array([], dtype=float)