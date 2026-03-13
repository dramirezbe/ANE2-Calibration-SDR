"""Integration tests for the SDR-side node-calibration facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from libs.data_request import CampaignParams, ConfigParams, ScheduleParams
from libs.node_calibration import (
    CalibrationServiceConfig,
    NodeCalibrationService,
    configuration_from_campaign_params,
)


SDR_REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_REPO_ROOT = (
    Path(__file__).resolve().parents[2] / "MeasurementCalibration"
).resolve()
TEST_CAMPAIGN_DIR = MEASUREMENT_REPO_ROOT / "data" / "campaigns" / "test-calibration"
LOCAL_MODEL_DIR = SDR_REPO_ROOT / "models" / "stable"


def _parse_bool(value: object) -> bool:
    """Return a stable boolean from CSV metadata values."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as boolean")


def _build_campaign_params() -> CampaignParams:
    """Return one ``CampaignParams`` instance from the checked-in metadata CSV."""

    metadata_frame = pd.read_csv(TEST_CAMPAIGN_DIR / "metadata.csv")
    metadata_row = metadata_frame.iloc[0]
    acquisition_interval_s = int(float(metadata_row["acquisition_freq_minutes"]) * 60.0)
    return CampaignParams(
        name=str(metadata_row["campaign_label"]),
        schedule=ScheduleParams(
            start_date=str(metadata_row["start_date"]),
            end_date=str(metadata_row["stop_date"]),
            start_time=str(metadata_row["start_time"]),
            end_time=str(metadata_row["stop_time"]),
            interval_seconds=acquisition_interval_s,
        ),
        config=ConfigParams(
            rbw=str(float(metadata_row["rbw_kHz"]) * 1.0e3),
            span=int(float(metadata_row["span_MHz"]) * 1.0e6),
            antenna="unknown",
            lna_gain=int(metadata_row["lna_gain_dB"]),
            vga_gain=int(metadata_row["vga_gain_dB"]),
            antenna_amp=_parse_bool(metadata_row["antenna_amp"]),
            center_freq_hz=int(float(metadata_row["central_freq_MHz"]) * 1.0e6),
            sample_rate_hz=int(metadata_row["sample_rate_hz"]),
            centerFrequency=int(float(metadata_row["central_freq_MHz"]) * 1.0e6),
        ),
        pxx_len=4096,
    )


def _load_campaign_frames() -> dict[str, pd.DataFrame]:
    """Load a compact subset of the checked-in test campaign as DataFrames."""

    sensor_ids = ("Node1", "Node2", "Node3", "Node10")
    frames: dict[str, pd.DataFrame] = {}
    for sensor_id in sensor_ids:
        frame = pd.read_csv(TEST_CAMPAIGN_DIR / f"{sensor_id}.csv").head(2).copy()
        frame["pxx"] = frame["pxx"].apply(json.loads)
        frames[sensor_id] = frame
    return frames


@dataclass(frozen=True)
class _FakeDataRequest:
    """Minimal ``DataRequest`` stub used to exercise the adapter surface."""

    campaign_params: CampaignParams
    campaign_frames: dict[str, dict[str, pd.DataFrame]]

    def get_campaign_params(self, campaign_id: int) -> CampaignParams:
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
    assert len(node1_result.calibrated_frame["calibrated_pxx"].iloc[0]) == 4096
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
