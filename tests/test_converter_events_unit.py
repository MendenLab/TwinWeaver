"""Tests that ConverterEvents respects DataSplitterEventsOption.unit_length_to_sample.

When the DataSplitterEvents is configured with a ``unit_length_to_sample`` that
differs from ``config.delta_time_unit``, the TTE prompt must:

1. Express the delta-time value in the *splitter's* unit, **not** the config's.
2. Use the *splitter's* unit name in the prompt text (e.g. "days" instead of
   "weeks").
"""

import pandas as pd

from twinweaver.common.config import Config
from twinweaver.common.data_manager import DataManager
from twinweaver.instruction.converter_events import ConverterEvents
from twinweaver.instruction.converter_instruction import ConverterInstruction
from twinweaver.instruction.data_splitter_events import (
    DataSplitterEvents,
    DataSplitterEventsOption,
)
from twinweaver.instruction.data_splitter import DataSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_option(
    delta: pd.Timedelta,
    unit: str = None,
    category_name: str = "death",
    category: str = "death",
    occurred: bool = True,
    censored=None,
) -> DataSplitterEventsOption:
    """Create a minimal DataSplitterEventsOption for unit-testing the converter."""
    split_date = pd.Timestamp("2020-01-01")
    return DataSplitterEventsOption(
        events_until_split=pd.DataFrame(),
        constant_data=pd.DataFrame(),
        event_occurred=occurred,
        event_censored=censored,
        observation_end_date=split_date + delta,
        split_date_included_in_input=split_date,
        sampled_category=category,
        sampled_category_name=category_name,
        lot_date=split_date,
        unit_length_to_sample=unit,
    )


def _make_converter(config: Config) -> ConverterEvents:
    """Create a ConverterEvents instance with the given config."""
    return ConverterEvents(
        config=config,
        constant_description=pd.DataFrame(),
        nr_tokens_budget_total=4096,
    )


# ---------------------------------------------------------------------------
# Unit-level tests on ConverterEvents._generate_prompt
# ---------------------------------------------------------------------------


class TestConverterEventsRespectsUnit:
    """ConverterEvents._generate_prompt must honour unit_length_to_sample."""

    def test_days_unit_overrides_config_weeks(self):
        """Splitter uses days while config uses weeks → prompt should say 'days'."""
        cfg = Config()
        cfg.seed = 42
        assert cfg.delta_time_unit == "weeks"  # default is weeks

        converter = _make_converter(cfg)
        option = _make_option(delta=pd.Timedelta(days=14), unit="days")

        prompt, delta_numeric = converter._generate_prompt(option)

        # 14 days expressed in *days* → 14
        assert "14" in prompt
        assert "days" in prompt
        # Make sure the config unit is NOT used for the numeric value
        # 14 days / 7 = 2 weeks — "2" could appear coincidentally, so check
        # the prompt text says "days" not "weeks" in the mid-section
        assert "days from the last clinical visit" in prompt

    def test_hours_unit_overrides_config_weeks(self):
        """Splitter uses hours while config uses weeks → prompt should say 'hours'."""
        cfg = Config()
        cfg.seed = 42
        converter = _make_converter(cfg)
        option = _make_option(delta=pd.Timedelta(hours=48), unit="hours")

        prompt, delta_numeric = converter._generate_prompt(option)

        assert "48" in prompt
        assert "hours" in prompt
        assert "hours from the last clinical visit" in prompt

    def test_minutes_unit_overrides_config_weeks(self):
        """Splitter uses minutes while config uses weeks."""
        cfg = Config()
        cfg.seed = 42
        converter = _make_converter(cfg)
        option = _make_option(delta=pd.Timedelta(minutes=120), unit="minutes")

        prompt, delta_numeric = converter._generate_prompt(option)

        assert "120" in prompt
        assert "minutes" in prompt

    def test_weeks_unit_matches_config_weeks(self):
        """Splitter uses weeks and so does config — should work identically."""
        cfg = Config()
        cfg.seed = 42
        converter = _make_converter(cfg)
        option = _make_option(delta=pd.Timedelta(weeks=4), unit="weeks")

        prompt, delta_numeric = converter._generate_prompt(option)

        assert "4" in prompt
        assert "weeks" in prompt
        assert abs(delta_numeric - 4.0) < 0.01

    def test_none_unit_falls_back_to_config(self):
        """When unit_length_to_sample is None, the config unit is used (legacy behaviour)."""
        cfg = Config()
        cfg.seed = 42
        cfg.set_delta_time_unit("days", unit_sing="day")

        converter = _make_converter(cfg)
        option = _make_option(delta=pd.Timedelta(days=14), unit=None)

        prompt, delta_numeric = converter._generate_prompt(option)

        assert "14" in prompt
        assert "days" in prompt

    def test_numeric_value_is_correct_days(self):
        """Delta numeric should reflect the splitter's unit, not the config's."""
        cfg = Config()
        cfg.seed = 42
        converter = _make_converter(cfg)

        # 21 days = 3 weeks
        option = _make_option(delta=pd.Timedelta(days=21), unit="days")
        _, delta_numeric = converter._generate_prompt(option)
        assert abs(delta_numeric - 21.0) < 0.01

    def test_numeric_value_is_correct_weeks(self):
        """When splitter unit is weeks, 21 days → 3 weeks."""
        cfg = Config()
        cfg.seed = 42
        converter = _make_converter(cfg)

        option = _make_option(delta=pd.Timedelta(days=21), unit="weeks")
        _, delta_numeric = converter._generate_prompt(option)
        assert abs(delta_numeric - 3.0) < 0.01

    def test_forward_conversion_uses_split_unit(self):
        """forward_conversion should propagate the unit through _generate_prompt."""
        cfg = Config()
        cfg.seed = 42
        converter = _make_converter(cfg)

        option = _make_option(delta=pd.Timedelta(days=7), unit="days")
        prompt, target, meta = converter.forward_conversion(option)

        # Prompt should express time in days (7), not weeks (1)
        assert "7" in prompt
        assert "days" in prompt
        # Meta should carry the numeric delta in days
        assert abs(meta["delta_time_numeric"] - 7.0) < 0.01


# ---------------------------------------------------------------------------
# Integration test: DataSplitterEvents propagates unit to option
# ---------------------------------------------------------------------------


class TestDataSplitterEventsPopulatesUnit:
    """DataSplitterEvents.get_splits_from_patient must set unit_length_to_sample."""

    def test_unit_propagated_to_options(self, mock_config, sample_data):
        """Options created by the splitter carry the correct unit."""
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

        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
        dm.infer_var_types()

        # Create splitter with unit_length_to_sample="days"
        splitter = DataSplitterEvents(
            dm,
            config=mock_config,
            max_length_to_sample=pd.Timedelta(days=90),
            min_length_to_sample=pd.Timedelta(days=1),
            max_split_length_after_split_event=pd.Timedelta(days=90),
            unit_length_to_sample="days",
        )
        splitter.setup_variables()

        patient_data = dm.get_patient_data("p0")
        groups = splitter.get_splits_from_patient(
            patient_data,
            max_nr_samples_per_split=2,
        )

        # At least one group should be generated
        assert len(groups) > 0

        for group in groups:
            for option in group.events_options:
                assert option.unit_length_to_sample == "days"


# ---------------------------------------------------------------------------
# End-to-end: config says "minutes", splitter says "days" → prompt says "days"
# ---------------------------------------------------------------------------


class TestEndToEndUnitOverride:
    """Full pipeline: splitter unit overrides config unit in generated prompt."""

    def test_events_prompt_uses_splitter_unit_not_config(self, mock_config, sample_data):
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

        # Set config to "minutes" — this should NOT appear in events prompt
        mock_config.set_delta_time_unit("minutes", unit_sing="minute")

        dm = DataManager(config=mock_config)
        dm.load_indication_data(df_events, df_constant, df_constant_desc)
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
        dm.infer_var_types()

        # Splitter uses "days"
        splitter_events = DataSplitterEvents(
            dm,
            config=mock_config,
            max_length_to_sample=pd.Timedelta(days=90),
            min_length_to_sample=pd.Timedelta(days=1),
            max_split_length_after_split_event=pd.Timedelta(days=90),
            unit_length_to_sample="days",
        )
        splitter_events.setup_variables()

        data_splitter = DataSplitter(data_splitter_events=splitter_events)

        converter = ConverterInstruction(
            nr_tokens_budget_total=4096,
            config=mock_config,
            dm=dm,
            variable_stats=None,
        )

        patient_data = dm.get_patient_data("p0")
        f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

        assert f_splits is None  # No forecasting splitter configured
        assert e_splits is not None and len(e_splits) > 0

        # Pick the first events group that has options
        event_group = None
        for eg in e_splits:
            if len(eg) > 0:
                event_group = eg
                break
        assert event_group is not None, "No event splits generated"

        result = converter.forward_conversion(
            forecasting_splits=None,
            event_splits=event_group,
        )

        instruction = result["instruction"]

        # The events task prompt should say "days from the last clinical visit",
        # NOT "minutes from the last clinical visit".
        assert "days from the last clinical visit" in instruction
        assert "minutes from the last clinical visit" not in instruction
