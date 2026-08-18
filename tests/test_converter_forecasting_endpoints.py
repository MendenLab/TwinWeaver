"""Tests for selecting which endpoints (forecasting variables) get converted.

A single forecasting split can carry many endpoints (e.g. 15 clinical endpoints sampled at one
split date). These tests cover selecting a subset of those endpoints at prompt build time, both
for training (``forward_conversion``) and for inference (``forward_conversion_inference``).
"""

import numpy as np
import pandas as pd
import pytest

from twinweaver.common.config import Config
from twinweaver.common.data_manager import DataManager
from twinweaver.instruction.converter_forecasting import ConverterForecasting
from twinweaver.instruction.converter_instruction import ConverterInstruction
from twinweaver.instruction.data_splitter_forecasting import (
    DataSplitterForecastingOption,
)

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

_SPLIT_DATE = pd.Timestamp("2020-03-01")
_NR_ENDPOINTS = 15

#: descriptive names are what appears in the prompt / target text
_ENDPOINTS = [f"endpoint_{idx:02d}" for idx in range(_NR_ENDPOINTS)]
_DESCRIPTIVE = {endpoint: endpoint.replace("_", " ") for endpoint in _ENDPOINTS}


def _make_config():
    """Create a Config wired for the synthetic multi-endpoint timeline."""

    cfg = Config()
    cfg.seed = 42
    cfg.split_event_category = "lot"
    cfg.event_category_forecast = ["lab"]
    cfg.constant_columns_to_use = ["gender"]
    return cfg


def _make_events(cfg, day_offsets):
    """Build events with one measurement per endpoint at each of the given day offsets."""

    rows = []
    for day in day_offsets:
        for idx, endpoint in enumerate(_ENDPOINTS):
            rows.append(
                {
                    cfg.date_col: _SPLIT_DATE + pd.Timedelta(days=day),
                    cfg.event_category_col: "lab",
                    cfg.event_name_col: endpoint,
                    cfg.event_value_col: str(round(10.0 + idx + day / 100.0, 2)),
                    cfg.event_descriptive_name_col: _DESCRIPTIVE[endpoint],
                    cfg.source_col: "events",
                    cfg.meta_data_col: pd.NA,
                }
            )
    return pd.DataFrame(rows)


def _make_option(cfg):
    """Create a forecasting split option carrying all 15 endpoints."""

    return DataSplitterForecastingOption(
        events_until_split=_make_events(cfg, [-30, 0]),
        target_events_after_split=_make_events(cfg, [21, 42]),
        constant_data=pd.DataFrame({cfg.patient_id_col: ["p_test"], "gender": ["female"]}),
        split_date_included_in_input=_SPLIT_DATE,
        sampled_variables=list(_ENDPOINTS),
        lot_date=_SPLIT_DATE,
    )


def _make_dm(cfg):
    """Create a DataManager stub providing the variable types the converters look up."""

    dm = DataManager.__new__(DataManager)
    dm.config = cfg
    dm.variable_types = {endpoint: "numeric" for endpoint in _ENDPOINTS}
    dm.data_frames = {}
    dm.all_patientids = ["p_test"]
    return dm


def _make_converter(cfg):
    """Create a ConverterForecasting for the synthetic setup."""

    return ConverterForecasting(
        constant_description=pd.DataFrame(),
        nr_tokens_budget_total=4096,
        config=cfg,
        dm=_make_dm(cfg),
    )


def _prompted_endpoints(prompt):
    """Extract the descriptive endpoint names which the prompt asks for."""

    return {
        endpoint for endpoint, descriptive in _DESCRIPTIVE.items() if f"\t{descriptive} the future weeks " in prompt
    }


# ────────────────────────────────────────────────────────────────────────────
# All endpoints at once
# ────────────────────────────────────────────────────────────────────────────


def test_forward_conversion_handles_all_endpoints():
    """Verify that a 15-endpoint split is fully converted into prompt, target and metadata."""

    cfg = _make_config()
    converter = _make_converter(cfg)

    prompt, target, meta = converter.forward_conversion(_make_option(cfg))

    assert _prompted_endpoints(prompt) == set(_ENDPOINTS)
    assert set(meta["future_prediction_time_per_variable"].keys()) == set(_ENDPOINTS)
    assert set(meta["variable_name_mapping"].keys()) == set(_ENDPOINTS)
    assert meta["target_data_processed"].shape[0] == _NR_ENDPOINTS * 2

    #: every endpoint must be asked for at both future time points (3 and 6 weeks)
    for endpoint in _ENDPOINTS:
        assert list(meta["future_prediction_time_per_variable"][endpoint]) == [3.0, 6.0]
        assert _DESCRIPTIVE[endpoint] in target

    #: the last observed value of every endpoint is available for the summarized row
    assert set(meta["last_observed_values"][cfg.event_name_col]) == set(_ENDPOINTS)


# ────────────────────────────────────────────────────────────────────────────
# Selecting a subset of endpoints
# ────────────────────────────────────────────────────────────────────────────


def test_forward_conversion_converts_only_selected_endpoints():
    """Verify that variables_to_convert restricts both the prompt and the target."""

    cfg = _make_config()
    converter = _make_converter(cfg)
    selected = ["endpoint_02", "endpoint_07"]

    prompt, target, meta = converter.forward_conversion(_make_option(cfg), variables_to_convert=selected)

    assert _prompted_endpoints(prompt) == set(selected)
    assert set(meta["future_prediction_time_per_variable"].keys()) == set(selected)
    assert set(meta["target_data_processed"][cfg.event_name_col].unique()) == set(selected)
    assert meta["target_data_processed"].shape[0] == len(selected) * 2

    #: the target text must not mention any of the other endpoints
    for endpoint in _ENDPOINTS:
        if endpoint not in selected:
            assert _DESCRIPTIVE[endpoint] not in target


def test_forward_conversion_accepts_a_single_endpoint_as_string():
    """Verify that a bare endpoint name is accepted as a convenience."""

    cfg = _make_config()
    converter = _make_converter(cfg)

    prompt, _, meta = converter.forward_conversion(_make_option(cfg), variables_to_convert="endpoint_04")

    assert _prompted_endpoints(prompt) == {"endpoint_04"}
    assert list(meta["future_prediction_time_per_variable"].keys()) == ["endpoint_04"]


def test_forward_conversion_yields_one_task_per_endpoint():
    """Verify that one multi-endpoint split can be converted into several narrower examples."""

    cfg = _make_config()
    converter = _make_converter(cfg)
    option = _make_option(cfg)

    for endpoint in _ENDPOINTS:
        prompt, _target, meta = converter.forward_conversion(option, variables_to_convert=[endpoint])
        assert _prompted_endpoints(prompt) == {endpoint}
        assert set(meta["target_data_processed"][cfg.event_name_col].unique()) == {endpoint}


def test_forward_conversion_does_not_mutate_the_split():
    """Verify that selecting a subset leaves the caller's split untouched."""

    cfg = _make_config()
    converter = _make_converter(cfg)
    option = _make_option(cfg)
    nr_target_rows_before = option.target_events_after_split.shape[0]

    converter.forward_conversion(option, variables_to_convert=["endpoint_00"])

    assert option.sampled_variables == list(_ENDPOINTS)
    assert option.target_events_after_split.shape[0] == nr_target_rows_before


def test_forward_conversion_warns_about_unknown_endpoints(caplog):
    """Verify that endpoints which are not part of the split are dropped with a warning."""

    cfg = _make_config()
    converter = _make_converter(cfg)

    with caplog.at_level("WARNING"):
        prompt, _, meta = converter.forward_conversion(
            _make_option(cfg), variables_to_convert=["endpoint_01", "not_a_real_endpoint"]
        )

    assert _prompted_endpoints(prompt) == {"endpoint_01"}
    assert list(meta["future_prediction_time_per_variable"].keys()) == ["endpoint_01"]
    assert any("not_a_real_endpoint" in record.message for record in caplog.records)


def test_forward_conversion_raises_when_no_endpoint_matches():
    """Verify that an entirely unmatched selection fails loudly instead of producing a bad prompt."""

    cfg = _make_config()
    converter = _make_converter(cfg)

    with pytest.raises(ValueError, match="variables_to_convert"):
        converter.forward_conversion(_make_option(cfg), variables_to_convert=["not_a_real_endpoint"])


# ────────────────────────────────────────────────────────────────────────────
# Inference path
# ────────────────────────────────────────────────────────────────────────────


def test_forward_conversion_inference_uses_descriptive_names_for_all_endpoints():
    """Verify that a 15-endpoint inference prompt resolves every descriptive name."""

    cfg = _make_config()
    converter = _make_converter(cfg)
    future_weeks = {endpoint: [4, 8, 12] for endpoint in _ENDPOINTS}

    prompt, meta = converter.forward_conversion_inference(_make_option(cfg), future_weeks_per_variable=future_weeks)

    assert _prompted_endpoints(prompt) == set(_ENDPOINTS)
    assert meta["variable_name_mapping"] == _DESCRIPTIVE
    for endpoint in _ENDPOINTS:
        assert len(meta["dates_per_variable"][endpoint]) == 3


def test_forward_conversion_inference_warns_about_unresolvable_endpoints(caplog):
    """
    Verify that an endpoint which cannot be resolved to a descriptive name is reported.

    Without a descriptive name the prompt asks for the raw internal name, which the reverse
    conversion cannot map back to a known event - it used to happen silently.
    """

    cfg = _make_config()
    converter = _make_converter(cfg)

    with caplog.at_level("WARNING"):
        converter.forward_conversion_inference(
            _make_option(cfg),
            future_weeks_per_variable={"endpoint_00": [4], "not_a_real_endpoint": [4]},
        )

    assert any("not_a_real_endpoint" in record.message for record in caplog.records)


# ────────────────────────────────────────────────────────────────────────────
# ConverterInstruction pass-through
# ────────────────────────────────────────────────────────────────────────────


def _make_instruction_converter(cfg):
    """Create a ConverterInstruction for the synthetic multi-endpoint setup."""

    dm = _make_dm(cfg)
    dm.unique_events = pd.DataFrame(
        {
            cfg.event_name_col: _ENDPOINTS,
            cfg.event_descriptive_name_col: [_DESCRIPTIVE[endpoint] for endpoint in _ENDPOINTS],
            cfg.event_category_col: ["lab"] * _NR_ENDPOINTS,
        }
    )
    dm.data_frames = {"constant_description": pd.DataFrame({"variable": ["gender"], "comment": ["gender of patient"]})}
    return ConverterInstruction(nr_tokens_budget_total=4096, config=cfg, dm=dm)


def test_converter_instruction_selects_forecasting_endpoints():
    """Verify that ConverterInstruction forwards forecasting_variables_to_convert."""

    cfg = _make_config()
    converter = _make_instruction_converter(cfg)
    selected = ["endpoint_03", "endpoint_11"]

    np.random.seed(42)
    result = converter.forward_conversion(
        forecasting_splits=[_make_option(cfg)],
        event_splits=[],
        override_mode_to_select_forecasting="forecasting",
        forecasting_variables_to_convert=selected,
    )

    assert _prompted_endpoints(result["instruction"]) == set(selected)
    for endpoint in _ENDPOINTS:
        if endpoint not in selected:
            assert _DESCRIPTIVE[endpoint] not in result["answer"]


def test_converter_instruction_inference_selects_forecasting_endpoints():
    """Verify that the inference path can select one endpoint out of a shared horizon spec."""

    cfg = _make_config()
    converter = _make_instruction_converter(cfg)
    future_weeks = {endpoint: [4, 8] for endpoint in _ENDPOINTS}

    result = converter.forward_conversion_inference(
        forecasting_split=_make_option(cfg),
        forecasting_future_weeks_per_variable=future_weeks,
        forecasting_variables_to_convert=["endpoint_05"],
    )

    assert _prompted_endpoints(result["instruction"]) == {"endpoint_05"}
    assert result["answer"] is None

    #: the caller's dict must not be modified
    assert set(future_weeks.keys()) == set(_ENDPOINTS)


def test_converter_instruction_inference_raises_when_no_endpoint_matches():
    """Verify that an unmatched inference selection fails loudly."""

    cfg = _make_config()
    converter = _make_instruction_converter(cfg)

    with pytest.raises(ValueError, match="forecasting_variables_to_convert"):
        converter.forward_conversion_inference(
            forecasting_split=_make_option(cfg),
            forecasting_future_weeks_per_variable={"endpoint_00": [4]},
            forecasting_variables_to_convert=["not_a_real_endpoint"],
        )


def test_converter_instruction_inference_requires_future_weeks():
    """Verify that selecting endpoints without a horizon specification is an error."""

    cfg = _make_config()
    converter = _make_instruction_converter(cfg)

    with pytest.raises(ValueError, match="forecasting_future_weeks_per_variable"):
        converter.forward_conversion_inference(
            forecasting_split=_make_option(cfg),
            forecasting_future_weeks_per_variable=None,
            forecasting_variables_to_convert=["endpoint_00"],
        )
