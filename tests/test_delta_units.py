"""Tests for extended delta_time_unit support.

Covers:
- Config.set_delta_time_unit with hours, minutes, seconds
- Fractional days support
- ConverterBase _time_divisor mapping for all units
- _delta_to_timedelta round-trip precision
- DataSplitterEvents unit_length_to_sample with new units
- End-to-end forward conversion with non-default units
"""

import pytest
import pandas as pd

from twinweaver.common.config import Config
from twinweaver.common.data_manager import DataManager
from twinweaver.instruction.converter_instruction import ConverterInstruction
from twinweaver.instruction.data_splitter_events import DataSplitterEvents
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting
from twinweaver.instruction.data_splitter import DataSplitter
from twinweaver.pretrain.converter_pretrain import ConverterPretrain


# ── Config unit tests ──────────────────────────────────────────────────────


class TestConfigSetDeltaTimeUnit:
    """Tests for Config.set_delta_time_unit with all supported units."""

    def test_set_hours(self):
        cfg = Config()
        cfg.set_delta_time_unit("hours", unit_sing="hour")
        assert cfg.delta_time_unit == "hours"
        assert "hours" in cfg.event_day_text
        assert "hour" in cfg.forecasting_fval_prompt_start

    def test_set_minutes(self):
        cfg = Config()
        cfg.set_delta_time_unit("minutes", unit_sing="minute")
        assert cfg.delta_time_unit == "minutes"
        assert "minutes" in cfg.event_day_text
        assert "minute" in cfg.forecasting_fval_prompt_start

    def test_set_seconds(self):
        cfg = Config()
        cfg.set_delta_time_unit("seconds", unit_sing="second")
        assert cfg.delta_time_unit == "seconds"
        assert "seconds" in cfg.event_day_text
        assert "second" in cfg.forecasting_fval_prompt_start

    def test_set_hours_parenthesised(self):
        cfg = Config()
        cfg.set_delta_time_unit("hour(s)")
        assert cfg.delta_time_unit == "hour(s)"
        assert "hour(s)" in cfg.event_day_text

    def test_set_minutes_parenthesised(self):
        cfg = Config()
        cfg.set_delta_time_unit("minute(s)")
        assert cfg.delta_time_unit == "minute(s)"

    def test_set_seconds_parenthesised(self):
        cfg = Config()
        cfg.set_delta_time_unit("second(s)")
        assert cfg.delta_time_unit == "second(s)"

    def test_set_days_still_works(self):
        cfg = Config()
        cfg.set_delta_time_unit("days", unit_sing="day")
        assert cfg.delta_time_unit == "days"
        assert "days" in cfg.event_day_text

    def test_set_weeks_still_works(self):
        cfg = Config()
        cfg.set_delta_time_unit("weeks", unit_sing="week")
        assert cfg.delta_time_unit == "weeks"
        assert "weeks" in cfg.event_day_text

    def test_invalid_unit_raises(self):
        cfg = Config()
        with pytest.raises(AssertionError):
            cfg.set_delta_time_unit("fortnights")

    def test_invalid_singular_raises(self):
        cfg = Config()
        with pytest.raises(AssertionError):
            cfg.set_delta_time_unit("hours", unit_sing="fortnight")

    def test_singular_defaults_to_plural(self):
        """When unit_sing is None the plural form is used in prompts."""
        cfg = Config()
        cfg.set_delta_time_unit("hours")
        assert "hours" in cfg.forecasting_fval_prompt_start

    def test_class_constants_complete(self):
        """The class-level constant tuples must contain all expected entries."""
        assert "hours" in Config.VALID_DELTA_UNITS_PLURAL
        assert "minutes" in Config.VALID_DELTA_UNITS_PLURAL
        assert "seconds" in Config.VALID_DELTA_UNITS_PLURAL
        assert "hour" in Config.VALID_DELTA_UNITS_SINGULAR
        assert "minute" in Config.VALID_DELTA_UNITS_SINGULAR
        assert "second" in Config.VALID_DELTA_UNITS_SINGULAR


# ── ConverterBase _time_divisor tests ──────────────────────────────────────


class TestConverterBaseTimeDivisor:
    """Test that _time_divisor is set correctly for all unit variants."""

    @pytest.fixture()
    def _make_converter(self, mock_config, sample_data):
        """Factory that returns a ConverterPretrain for a given delta_time_unit."""
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]

        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()

        def _build(unit: str):
            mock_config.set_delta_time_unit(unit)
            return ConverterPretrain(config=mock_config, dm=dm)

        return _build

    @pytest.mark.parametrize(
        "unit, expected_divisor",
        [
            ("weeks", 7.0),
            ("week(s)", 7.0),
            ("days", 1.0),
            ("day(s)", 1.0),
            ("hours", 1.0 / 24.0),
            ("hour(s)", 1.0 / 24.0),
            ("minutes", 1.0 / (24.0 * 60.0)),
            ("minute(s)", 1.0 / (24.0 * 60.0)),
            ("seconds", 1.0 / (24.0 * 60.0 * 60.0)),
            ("second(s)", 1.0 / (24.0 * 60.0 * 60.0)),
        ],
    )
    def test_time_divisor_values(self, _make_converter, unit, expected_divisor):
        converter = _make_converter(unit)
        assert converter._time_divisor == pytest.approx(expected_divisor)

    def test_unsupported_unit_raises(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()

        # Bypass the Config assertion to force an unsupported unit
        mock_config.delta_time_unit = "fortnights"
        with pytest.raises(ValueError, match="Unsupported delta_time_unit"):
            ConverterPretrain(config=mock_config, dm=dm)


# ── _delta_to_timedelta precision tests ────────────────────────────────────


class TestDeltaToTimedelta:
    """Test that _delta_to_timedelta round-trips correctly."""

    @pytest.fixture()
    def converter_for_unit(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()

        def _build(unit: str):
            mock_config.set_delta_time_unit(unit)
            return ConverterPretrain(config=mock_config, dm=dm)

        return _build

    def test_weeks_rounds_to_days(self, converter_for_unit):
        conv = converter_for_unit("weeks")
        td = conv._delta_to_timedelta(3.0)  # 3 weeks = 21 days
        assert td == pd.Timedelta(days=21)
        # Fractional weeks: 2.5 weeks = 17.5 days → rounded to 18
        td2 = conv._delta_to_timedelta(2.5)
        assert td2 == pd.Timedelta(days=18)

    def test_days_rounds_to_days(self, converter_for_unit):
        conv = converter_for_unit("days")
        td = conv._delta_to_timedelta(5.0)
        assert td == pd.Timedelta(days=5)
        # Fractional day: 0.7 → round to 1
        td2 = conv._delta_to_timedelta(0.7)
        assert td2 == pd.Timedelta(days=1)

    def test_hours_preserves_sub_day(self, converter_for_unit):
        conv = converter_for_unit("hours")
        td = conv._delta_to_timedelta(6.0)  # 6 hours
        assert td == pytest.approx(pd.Timedelta(hours=6), abs=pd.Timedelta(seconds=1))

    def test_minutes_preserves_sub_day(self, converter_for_unit):
        conv = converter_for_unit("minutes")
        td = conv._delta_to_timedelta(90.0)  # 90 min = 1.5 hours
        assert td == pytest.approx(pd.Timedelta(minutes=90), abs=pd.Timedelta(seconds=1))

    def test_seconds_preserves_sub_day(self, converter_for_unit):
        conv = converter_for_unit("seconds")
        td = conv._delta_to_timedelta(3600.0)  # 3600 s = 1 hour
        # Allow tiny floating-point rounding (nanosecond level)
        assert abs(td - pd.Timedelta(hours=1)) < pd.Timedelta(microseconds=1)


# ── Fractional days in text conversion ─────────────────────────────────────


class TestFractionalDays:
    """Fractional day deltas should appear in generated text."""

    @pytest.fixture()
    def converter_days(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()

        mock_config.set_delta_time_unit("days")
        converter = ConverterPretrain(config=mock_config, dm=dm)
        return dm, converter

    def test_fractional_day_text(self, converter_days):
        """When events are <1 day apart the delta should be a fraction like '0.5'."""
        dm, converter = converter_days

        # Build a minimal 2-event DataFrame with 12 hours between events
        events = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00"]),
                "event_category": ["lab", "lab"],
                "event_name": ["glucose", "glucose"],
                "event_descriptive_name": ["glucose", "glucose"],
                "event_value": ["100", "110"],
                "source": ["events", "events"],
                "meta_data": [pd.NA, pd.NA],
            }
        )

        text = converter._get_event_string(events)
        # 12 h = 0.5 days
        assert "0.5" in text
        assert "days" in text


# ── DataSplitterEvents with new units ──────────────────────────────────────


class TestDataSplitterEventsUnits:
    """DataSplitterEvents should accept hours, minutes, seconds."""

    @pytest.fixture()
    def initialized_dm(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.split_event_category = "lot"
        mock_config.event_category_forecast = ["lab"]
        mock_config.event_category_events_prediction_with_naming = {
            "death": "death",
            "progression": "next progression",
        }
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
        dm.infer_var_types()
        return dm

    @pytest.mark.parametrize("unit", ["days", "hours", "minutes", "seconds"])
    def test_splitter_accepts_unit(self, initialized_dm, mock_config, unit):
        """DataSplitterEvents should instantiate without error for each unit."""
        splitter = DataSplitterEvents(
            initialized_dm,
            config=mock_config,
            max_length_to_sample=pd.Timedelta(weeks=104),
            min_length_to_sample=pd.Timedelta(weeks=1),
            unit_length_to_sample=unit,
            max_split_length_after_split_event=pd.Timedelta(days=90),
        )
        splitter.setup_variables()
        assert splitter.unit_length_to_sample == unit

    def test_splitter_unsupported_unit(self, initialized_dm, mock_config):
        splitter = DataSplitterEvents(
            initialized_dm,
            config=mock_config,
            max_length_to_sample=pd.Timedelta(weeks=104),
            min_length_to_sample=pd.Timedelta(weeks=1),
            unit_length_to_sample="fortnights",
            max_split_length_after_split_event=pd.Timedelta(days=90),
        )
        splitter.setup_variables()
        patient_data = initialized_dm.get_patient_data("p0")
        with pytest.raises(NotImplementedError, match="not implemented"):
            splitter.get_splits_from_patient(patient_data, max_nr_samples_per_split=1)


# ── End-to-end conversion with "days" unit ─────────────────────────────────


class TestEndToEndDaysUnit:
    """Integration: full forward conversion with delta_time_unit='days'."""

    @pytest.fixture()
    def setup_days(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.split_event_category = "lot"
        mock_config.event_category_forecast = ["lab"]
        mock_config.event_category_events_prediction_with_naming = {
            "death": "death",
            "progression": "next progression",
        }
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        mock_config.constant_birthdate_column = "birthyear"
        mock_config.set_delta_time_unit("days", unit_sing="day")

        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
        dm.infer_var_types()

        splitter_events = DataSplitterEvents(
            dm,
            config=mock_config,
            max_length_to_sample=pd.Timedelta(days=730),
            min_length_to_sample=pd.Timedelta(days=1),
            unit_length_to_sample="days",
            max_split_length_after_split_event=pd.Timedelta(days=90),
        )
        splitter_events.setup_variables()

        splitter_forecast = DataSplitterForecasting(
            data_manager=dm,
            config=mock_config,
            max_forecasted_trajectory_length=pd.Timedelta(days=90),
            max_split_length_after_split_event=pd.Timedelta(days=90),
        )
        splitter_forecast.setup_statistics()

        data_splitter = DataSplitter(splitter_events, splitter_forecast)

        converter = ConverterInstruction(
            nr_tokens_budget_total=4096,
            config=mock_config,
            dm=dm,
            variable_stats=splitter_forecast.variable_stats,
        )
        return dm, data_splitter, converter

    def test_forward_conversion_uses_days(self, setup_days):
        dm, data_splitter, converter = setup_days
        patient_data = dm.get_patient_data("p0")
        f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

        result = converter.forward_conversion(
            forecasting_splits=f_splits[0],
            event_splits=e_splits[0],
            override_mode_to_select_forecasting="both",
        )

        instruction = result["instruction"]
        # "weeks" should NOT appear in the temporal text, "days" should
        assert "days later" in instruction
        assert "weeks later" not in instruction
        # Demographic section should still be present
        assert "Starting with demographic data:" in instruction


# ── End-to-end conversion with "hours" unit ────────────────────────────────


class TestEndToEndHoursUnit:
    """Integration: full forward conversion with delta_time_unit='hours'."""

    @pytest.fixture()
    def setup_hours(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.split_event_category = "lot"
        mock_config.event_category_forecast = ["lab"]
        mock_config.event_category_events_prediction_with_naming = {
            "death": "death",
            "progression": "next progression",
        }
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        mock_config.constant_birthdate_column = "birthyear"
        mock_config.set_delta_time_unit("hours", unit_sing="hour")

        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
        dm.infer_var_types()

        splitter_events = DataSplitterEvents(
            dm,
            config=mock_config,
            max_length_to_sample=pd.Timedelta(weeks=104),
            min_length_to_sample=pd.Timedelta(hours=1),
            unit_length_to_sample="hours",
            max_split_length_after_split_event=pd.Timedelta(days=90),
        )
        splitter_events.setup_variables()

        splitter_forecast = DataSplitterForecasting(
            data_manager=dm,
            config=mock_config,
            max_forecasted_trajectory_length=pd.Timedelta(days=90),
            max_split_length_after_split_event=pd.Timedelta(days=90),
        )
        splitter_forecast.setup_statistics()

        data_splitter = DataSplitter(splitter_events, splitter_forecast)

        converter = ConverterInstruction(
            nr_tokens_budget_total=4096,
            config=mock_config,
            dm=dm,
            variable_stats=splitter_forecast.variable_stats,
        )
        return dm, data_splitter, converter

    def test_forward_conversion_uses_hours(self, setup_hours):
        dm, data_splitter, converter = setup_hours
        patient_data = dm.get_patient_data("p0")
        f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

        result = converter.forward_conversion(
            forecasting_splits=f_splits[0],
            event_splits=e_splits[0],
            override_mode_to_select_forecasting="both",
        )

        instruction = result["instruction"]
        # The temporal text should use "hours", not "weeks" or "days"
        assert "hours later" in instruction
        assert "weeks later" not in instruction
        assert "Starting with demographic data:" in instruction


# ── Pretrain conversion with "days" unit ───────────────────────────────────


class TestPretrainDaysUnit:
    """Integration: pretrain forward conversion with delta_time_unit='days'."""

    @pytest.fixture()
    def setup_pretrain_days(self, mock_config, sample_data):
        df_events, df_constant, df_constant_desc = sample_data
        mock_config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]
        mock_config.constant_birthdate_column = "birthyear"
        mock_config.set_delta_time_unit("days", unit_sing="day")

        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()

        converter = ConverterPretrain(config=mock_config, dm=dm)
        return dm, converter

    def test_pretrain_forward_uses_days(self, setup_pretrain_days):
        dm, converter = setup_pretrain_days
        patient_data = dm.get_patient_data("p0")

        result = converter.forward_conversion(patient_data["events"], patient_data["constant"])
        text = result["text"]

        assert "days later" in text
        assert "weeks later" not in text
        assert "Starting with demographic data:" in text

    def test_pretrain_roundtrip_days(self, setup_pretrain_days):
        """Forward → reverse should preserve data at day-level precision for days unit."""
        dm, converter = setup_pretrain_days
        patient_data = dm.get_patient_data("p0")

        forward_result = converter.forward_conversion(patient_data["events"], patient_data["constant"])
        text = forward_result["text"]
        meta = forward_result["meta"]

        reverse_result = converter.reverse_conversion(
            text=text, data_manager=dm, init_date=meta["events"]["date"].min()
        )
        diff = converter.get_difference_in_event_dataframes(meta["events"], reverse_result["events"], skip_genetic=True)
        assert diff.shape[0] == 0, f"Found differences in roundtrip conversion:\n{diff}"
