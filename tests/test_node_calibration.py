"""Integration tests for the SDR-side node-calibration facade.

The test fixtures are intentionally synthetic so the SDR fork remains fully
self-contained. The deployed artifact is loaded from ``models/stable`` inside
the fork, while PSD observations are generated in-memory with deterministic
frequency bounds that lie inside the artifact support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from libs.node_calibration import (
    CalibrationServiceConfig,
    NodeCalibrationService,
    configuration_from_campaign_params,
)


SDR_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = SDR_REPO_ROOT / "models" / "stable"


@dataclass(frozen=True)
class _ScheduleParams:
    """Minimal schedule payload compatible with ``configuration_from_campaign_params``."""

    start_date: str
    end_date: str
    start_time: str
    end_time: str
    interval_seconds: int


@dataclass(frozen=True)
class _ConfigParams:
    """Minimal configuration payload compatible with ``configuration_from_campaign_params``."""

    rbw: str
    span: int
    antenna: str
    lna_gain: int
    vga_gain: int
    antenna_amp: bool
    center_freq_hz: int
    sample_rate_hz: int
    centerFrequency: int


@dataclass(frozen=True)
class _CampaignParams:
    """Minimal campaign payload that matches the SDR adapter contract."""

    name: str
    schedule: _ScheduleParams
    config: _ConfigParams
    pxx_len: int


def _build_campaign_params() -> _CampaignParams:
    """Return one stable ``CampaignParams`` instance within artifact support."""

    return _CampaignParams(
        name="synthetic-calibration",
        schedule=_ScheduleParams(
            start_date="2025-02-24T00:00:00.000Z",
            end_date="2025-02-24T00:00:00.000Z",
            start_time="10:30:00",
            end_time="14:00:00",
            interval_seconds=120,
        ),
        config=_ConfigParams(
            rbw="10000",
            span=20_000_000,
            antenna="unknown",
            lna_gain=40,
            vga_gain=62,
            antenna_amp=True,
            center_freq_hz=98_000_000,
            sample_rate_hz=20_000_000,
            centerFrequency=98_000_000,
        ),
        pxx_len=16,
    )


def _load_campaign_frames() -> dict[str, pd.DataFrame]:
    """Build deterministic synthetic PSD frames for four sensors."""

    frequency_bin_count = 16
    base_spectrum_db = np.linspace(-120.0, -80.0, frequency_bin_count, dtype=np.float64)
    sensor_offsets_db = {
        "Node1": 0.0,
        "Node2": 1.5,
        "Node3": -2.0,
        "Node10": 3.0,
    }

    frames: dict[str, pd.DataFrame] = {}
    for sensor_id, offset_db in sensor_offsets_db.items():
        first_row = (base_spectrum_db + offset_db).tolist()
        second_row = (base_spectrum_db + offset_db + 0.25).tolist()
        frames[sensor_id] = pd.DataFrame(
            [
                {
                    "id": 1,
                    "mac": "00:11:22:33:44:55",
                    "campaign_id": 205,
                    "pxx": first_row,
                    "start_freq_hz": 97_000_000.0,
                    "end_freq_hz": 99_000_000.0,
                },
                {
                    "id": 2,
                    "mac": "00:11:22:33:44:55",
                    "campaign_id": 205,
                    "pxx": second_row,
                    "start_freq_hz": 97_000_000.0,
                    "end_freq_hz": 99_000_000.0,
                },
            ]
        )
    return frames


@dataclass(frozen=True)
class _FakeDataRequest:
    """Minimal ``DataRequest`` stub used to exercise the adapter surface."""

    campaign_params: _CampaignParams
    campaign_frames: dict[str, dict[str, pd.DataFrame]]

    def get_campaign_params(self, campaign_id: int) -> _CampaignParams:
        """Return the preloaded campaign parameters regardless of id."""

        return self.campaign_params

    def load_campaigns_and_nodes(
        self,
        campaigns: dict[str, int],
        node_ids: list[int],
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Return the preloaded frames without contacting the remote API."""

        del campaigns
        del node_ids
        return self.campaign_frames


def test_configuration_from_campaign_params_maps_data_request_fields() -> None:
    """The adapter should convert SDR campaign params into model features."""

    configuration = configuration_from_campaign_params(_build_campaign_params())

    assert configuration.central_frequency_hz == 98_000_000.0
    assert configuration.span_hz == 20_000_000.0
    assert configuration.resolution_bandwidth_hz == 10_000.0
    assert configuration.acquisition_interval_s == 120.0
    assert configuration.antenna_amplifier_enabled is True


def test_calibrate_loaded_campaigns_uses_data_request_contract() -> None:
    """Campaign calibration should work from ``DataRequest``-shaped inputs."""

    service = NodeCalibrationService(
        CalibrationServiceConfig(
            model_dir=LOCAL_MODEL_DIR,
            skip_unknown_sensors=True,
        )
    )
    campaign_label = "test-calibration"
    frames = _load_campaign_frames()
    data_request = _FakeDataRequest(
        campaign_params=_build_campaign_params(),
        campaign_frames={campaign_label: frames},
    )

    results = service.calibrate_loaded_campaigns(
        data_request=data_request,
        campaigns={campaign_label: 205},
        node_ids=[1, 2, 3, 10],
    )
    campaign_result = results[campaign_label]

    assert set(campaign_result.calibrated_sensors) == {"Node1", "Node2", "Node3"}
    assert "Node10" in campaign_result.skipped_sensors

    node1_result = campaign_result.calibrated_sensors["Node1"]
    assert node1_result.calibrated_frame.shape[0] == 2
    assert "calibrated_pxx" in node1_result.calibrated_frame.columns
    assert "calibrated_variance_power2" in node1_result.calibrated_frame.columns
    assert len(node1_result.calibrated_frame["calibrated_pxx"].iloc[0]) == 16
    assert np.all(np.isfinite(node1_result.calibrated_power_db))
    assert node1_result.trust_summary.frequency_extrapolation_detected is False


def test_materialize_campaign_writes_measurement_calibration_schema(
    tmp_path: Path,
) -> None:
    """Materialization should emit ``metadata.csv`` plus per-node CSV files."""

    service = NodeCalibrationService(
        CalibrationServiceConfig(
            model_dir=LOCAL_MODEL_DIR,
        )
    )
    frames = _load_campaign_frames()

    materialized = service.materialize_campaign(
        campaign_label="test-calibration",
        campaign_id=205,
        node_frames=frames,
        campaign_params=_build_campaign_params(),
        output_root=tmp_path,
    )

    metadata_frame = pd.read_csv(materialized.metadata_csv_path)
    node_frame = pd.read_csv(materialized.saved_sensor_csv_paths["Node1"])

    assert materialized.output_dir == tmp_path / "test-calibration"
    assert metadata_frame.iloc[0]["campaign_label"] == "test-calibration"
    assert metadata_frame.iloc[0]["campaign_id"] == 205
    assert node_frame.columns.tolist()[0:4] == ["id", "mac", "campaign_id", "pxx"]
    assert node_frame["pxx"].iloc[0].startswith("[")
