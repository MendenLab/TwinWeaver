import pytest
import numpy as np
import pandas as pd
from twinweaver.common.data_manager import DataManager
from twinweaver.instruction.data_splitter_events import DataSplitterEvents
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting
from twinweaver.instruction.data_splitter import DataSplitter
from twinweaver.instruction.data_splitter_base import BaseDataSplitter


@pytest.fixture
def initialized_dm(mock_config, sample_data):
    """Fixture that returns a fully processed DataManager."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
    dm.infer_var_types()
    return dm


def test_splitter_forecasting_statistics(initialized_dm, mock_config):
    """Test that forecasting splitter can calculate statistics."""
    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm, config=mock_config, max_forecasted_trajectory_length=pd.Timedelta(days=90)
    )

    # This calculates R2, NRMSE etc. for the variables
    splitter_forecast.setup_statistics()

    assert splitter_forecast.variable_stats is not None
    assert not splitter_forecast.variable_stats.empty

    # Check that hemoglobin exists in stats
    stats = splitter_forecast.variable_stats
    assert "hemoglobin_-_718-7" in stats["event_name"].values

    assert stats.shape[0] == 1  # Only one lab variable in test data
    hemoglobin_stats = stats[stats["event_name"] == "hemoglobin_-_718-7"].iloc[0]
    assert hemoglobin_stats["score_log_nrmse_n_samples"] == pytest.approx(2.2216119558656935)  # Manual calc
    assert hemoglobin_stats["mean_without_outliers"] == pytest.approx(13.149285714285712)  # Manual calc
    assert hemoglobin_stats["std_without_outliers"] == pytest.approx(0.6813690450190734)  # Manual calc
    assert hemoglobin_stats["num_samples"] == 14  # Manual count


def test_get_splits_from_patient(initialized_dm, mock_config):
    """Test generating splits for a single patient."""
    # Setup Splitters
    splitter_events = DataSplitterEvents(
        initialized_dm,
        config=mock_config,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_events.setup_variables()

    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm,
        config=mock_config,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_forecast.setup_statistics()

    data_splitter = DataSplitter(splitter_events, splitter_forecast)

    # Get Patient Data
    patient_data = initialized_dm.get_patient_data("p0")

    # Generate Splits
    forecasting_splits, events_splits, ref_dates = data_splitter.get_splits_from_patient_with_target(
        patient_data, max_num_splits_per_split_event=1
    )

    # Assertions
    assert len(forecasting_splits) == 1  # Only one split due to max_num_splits_per_split_event=1
    assert len(events_splits) == 1  # Only one split due to max_num_splits_per_split_event=1
    assert len(forecasting_splits) == len(events_splits)

    # Check structure of a split
    f_split = forecasting_splits[0][0]  # First group, first option
    e_split = events_splits[0][0]

    assert f_split.events_until_split is not None
    assert f_split.target_events_after_split is not None
    assert e_split.split_date_included_in_input == f_split.split_date_included_in_input
    assert e_split.events_until_split.shape == f_split.events_until_split.shape
    assert e_split.events_until_split["date"].unique().tolist() == f_split.events_until_split["date"].unique().tolist()
    assert e_split.constant_data["patientid"].iloc[0] == "p0"  # Constant data matches
    assert f_split.constant_data["patientid"].iloc[0] == "p0"
    assert e_split.lot_date == f_split.lot_date

    # Check specifics of e_split - all calculated manually given the random seed and sample data
    assert e_split.event_censored is None
    assert e_split.event_occurred
    assert e_split.observation_end_date == pd.Timestamp("2017-05-17 00:00:00")
    assert e_split.sampled_category == "death"

    # Check specifics of f_split - all calculated manually given the random seed and sample data
    assert f_split.sampled_variables.tolist() == ["hemoglobin_-_718-7"]
    assert f_split.target_events_after_split.shape[0] == 4  # 4 hemoglobin measurements after split
    assert f_split.target_events_after_split["date"].min() == pd.Timestamp("2015-06-17 00:00:00")
    assert f_split.target_events_after_split["date"].max() == pd.Timestamp("2015-08-19 00:00:00")


def test_inference_split(initialized_dm, mock_config):
    """Test generating an inference split (last date)."""
    splitter_events = DataSplitterEvents(
        initialized_dm,
        config=mock_config,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
    )
    splitter_events.setup_variables()
    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm, config=mock_config, max_forecasted_trajectory_length=pd.Timedelta(days=90)
    )

    data_splitter = DataSplitter(splitter_events, splitter_forecast)
    patient_data = initialized_dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data, inference_type="both", forecasting_override_variables_to_predict=["hemoglobin_-_718-7"]
    )

    # Should use the very last date in the patient history
    last_date = patient_data["events"]["date"].max()

    # Manual calculations based on test data
    assert f_split.split_date_included_in_input == last_date
    assert f_split.target_events_after_split.empty  # Inference has no target
    assert f_split.sampled_variables == ["hemoglobin_-_718-7"]
    assert f_split.lot_date == "override"

    # Manually calculated - defaults to random selection since we didn't provide explicit override
    assert e_split.split_date_included_in_input == last_date
    assert e_split.sampled_category == "progression"
    assert e_split.observation_end_date == pd.Timestamp("2018-02-23 00:00:00")
    assert e_split.event_censored == "end_of_data"
    assert not e_split.event_occurred


# ────────────────────────────────────────────────────────────────────────────
# Tests for DataSplitter with individual (single) splitters
# ────────────────────────────────────────────────────────────────────────────


def test_data_splitter_requires_at_least_one_splitter():
    """Test that DataSplitter raises if neither splitter is provided."""
    with pytest.raises(ValueError, match="At least one"):
        DataSplitter()


def test_training_forecasting_only(initialized_dm, mock_config):
    """Test training splits when only the forecasting splitter is provided."""
    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm,
        config=mock_config,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_forecast.setup_statistics()

    data_splitter = DataSplitter(data_splitter_forecasting=splitter_forecast)

    patient_data = initialized_dm.get_patient_data("p0")
    forecasting_splits, events_splits, ref_dates = data_splitter.get_splits_from_patient_with_target(
        patient_data, max_num_splits_per_split_event=1
    )

    # Forecasting should be populated
    assert forecasting_splits is not None
    assert len(forecasting_splits) == 1

    # Events should be None since no events splitter was provided
    assert events_splits is None

    # Reference dates should still be available from the forecasting splitter
    assert ref_dates is not None
    assert not ref_dates.empty
    assert "date" in ref_dates.columns
    assert "split_date" in ref_dates.columns
    assert ref_dates.shape == (1, 2)
    assert ref_dates["date"].iloc[0] == pd.Timestamp("2015-05-27")
    assert ref_dates["split_date"].iloc[0] == pd.Timestamp("2015-05-06")

    # Validate forecasting split structure and content
    f_split = forecasting_splits[0][0]
    assert f_split.events_until_split is not None
    assert f_split.constant_data["patientid"].iloc[0] == "p0"
    assert f_split.events_until_split.shape == (23, 8)
    assert f_split.split_date_included_in_input == pd.Timestamp("2015-05-27")
    assert f_split.lot_date == pd.Timestamp("2015-05-06")
    assert f_split.sampled_variables == ["hemoglobin_-_718-7"]

    # Target events should exist and be after the split date
    assert not f_split.target_events_after_split.empty
    assert f_split.target_events_after_split.shape[0] == 4
    assert f_split.target_events_after_split["date"].min() == pd.Timestamp("2015-06-17")
    assert f_split.target_events_after_split["date"].max() == pd.Timestamp("2015-08-19")

    # Events in input must be <= split date
    assert (f_split.events_until_split["date"] <= f_split.split_date_included_in_input).all()
    assert f_split.events_until_split["date"].min() == pd.Timestamp("2015-04-19")

    # Constant data should contain expected columns
    assert "birthyear" in f_split.constant_data.columns
    assert "gender" in f_split.constant_data.columns
    assert f_split.constant_data.shape[0] == 1


def test_training_events_only(initialized_dm, mock_config):
    """Test training splits when only the events splitter is provided."""
    splitter_events = DataSplitterEvents(
        initialized_dm,
        config=mock_config,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_events.setup_variables()

    data_splitter = DataSplitter(data_splitter_events=splitter_events)

    patient_data = initialized_dm.get_patient_data("p0")
    forecasting_splits, events_splits, ref_dates = data_splitter.get_splits_from_patient_with_target(
        patient_data, max_num_splits_per_split_event=1
    )

    # Forecasting should be None since no forecasting splitter was provided
    assert forecasting_splits is None

    # Events should be populated
    assert events_splits is not None
    assert len(events_splits) == 1

    # Reference dates should be reconstructed from events splits
    assert ref_dates is not None
    assert not ref_dates.empty
    assert "date" in ref_dates.columns
    assert "split_date" in ref_dates.columns
    assert ref_dates.shape == (1, 2)
    assert ref_dates["date"].iloc[0] == pd.Timestamp("2015-05-27")
    assert ref_dates["split_date"].iloc[0] == pd.Timestamp("2015-05-06")

    # Validate events split structure and content
    e_split = events_splits[0][0]
    assert e_split.events_until_split is not None
    assert e_split.constant_data["patientid"].iloc[0] == "p0"
    assert e_split.events_until_split.shape == (23, 8)
    assert e_split.split_date_included_in_input == pd.Timestamp("2015-05-27")
    assert e_split.lot_date == pd.Timestamp("2015-05-06")
    # Category must be one of the configured event categories (or a backup thereof)
    expected_mapping = {"death": "death", "progression": "next progression"}
    assert e_split.sampled_category in list(expected_mapping.keys()) + list(
        mock_config.data_splitter_events_backup_category_mapping.values()
    )
    # Category name must match one of the configured descriptive names
    assert e_split.sampled_category_name in expected_mapping.values()

    # Event outcome must be boolean
    assert isinstance(e_split.event_occurred, bool)
    # Censoring should be None or one of the known censoring types
    assert e_split.event_censored in [None, "new_split_date_start", "end_of_data", "data_cutoff"]
    # Observation end date must be after the split date
    assert e_split.observation_end_date >= e_split.split_date_included_in_input

    # Events in input must be <= split date
    assert (e_split.events_until_split["date"] <= e_split.split_date_included_in_input).all()
    assert e_split.events_until_split["date"].min() == pd.Timestamp("2015-04-19")
    assert e_split.events_until_split["date"].max() == pd.Timestamp("2015-05-27")

    # Constant data integrity
    assert "birthyear" in e_split.constant_data.columns
    assert "gender" in e_split.constant_data.columns
    assert "histology" in e_split.constant_data.columns
    assert "smoking_history" in e_split.constant_data.columns
    assert e_split.constant_data.shape[0] == 1


def test_inference_forecasting_only(initialized_dm, mock_config):
    """Test inference split when only the forecasting splitter is provided."""
    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm, config=mock_config, max_forecasted_trajectory_length=pd.Timedelta(days=90)
    )

    data_splitter = DataSplitter(data_splitter_forecasting=splitter_forecast)
    patient_data = initialized_dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="forecasting",
        forecasting_override_variables_to_predict=["hemoglobin_-_718-7"],
    )

    last_date = patient_data["events"]["date"].max()

    # Forecasting split should be populated
    assert f_split is not None
    assert f_split.split_date_included_in_input == last_date
    assert f_split.split_date_included_in_input == pd.Timestamp("2016-05-13")
    assert f_split.sampled_variables == ["hemoglobin_-_718-7"]
    assert f_split.lot_date == "override"

    # Inference has no target data
    assert f_split.target_events_after_split.empty

    # Input events must cover the full patient history up to last date
    assert f_split.events_until_split.shape == (78, 8)
    assert (f_split.events_until_split["date"] <= f_split.split_date_included_in_input).all()
    assert f_split.events_until_split["date"].min() == pd.Timestamp("2015-04-19")
    assert f_split.events_until_split["date"].max() == pd.Timestamp("2016-05-13")

    # Constant data integrity
    assert f_split.constant_data["patientid"].iloc[0] == "p0"
    assert "birthyear" in f_split.constant_data.columns
    assert "gender" in f_split.constant_data.columns
    assert f_split.constant_data.shape[0] == 1

    # Events split should be None
    assert e_split is None


def test_inference_events_only(initialized_dm, mock_config):
    """Test inference split when only the events splitter is provided."""
    splitter_events = DataSplitterEvents(
        initialized_dm,
        config=mock_config,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
    )
    splitter_events.setup_variables()

    data_splitter = DataSplitter(data_splitter_events=splitter_events)
    patient_data = initialized_dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="events",
        events_override_category="death",
        events_override_observation_time_delta=pd.Timedelta(weeks=52),
    )

    last_date = patient_data["events"]["date"].max()

    # Forecasting split should be None
    assert f_split is None

    # Events split should be populated
    assert e_split is not None
    assert e_split.split_date_included_in_input == last_date
    assert e_split.split_date_included_in_input == pd.Timestamp("2016-05-13")
    assert e_split.sampled_category == "death"
    assert e_split.sampled_category_name == "death"

    # p0's last event is death itself; predicting death from last date with 52-week window:
    # death already occurred at last_date so looking forward finds nothing → censored end_of_data
    assert e_split.event_occurred is False
    assert e_split.event_censored == "end_of_data"
    assert e_split.observation_end_date == pd.Timestamp("2017-05-12")

    # Input events must cover the full patient history up to last date
    assert e_split.events_until_split.shape == (78, 8)
    assert (e_split.events_until_split["date"] <= e_split.split_date_included_in_input).all()
    assert e_split.events_until_split["date"].max() == pd.Timestamp("2016-05-13")

    # Constant data integrity
    assert e_split.constant_data["patientid"].iloc[0] == "p0"
    assert "birthyear" in e_split.constant_data.columns
    assert e_split.constant_data.shape[0] == 1


def test_inference_both_type_with_only_forecasting(initialized_dm, mock_config):
    """Test that inference_type='both' gracefully returns None for the missing splitter."""
    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm, config=mock_config, max_forecasted_trajectory_length=pd.Timedelta(days=90)
    )

    data_splitter = DataSplitter(data_splitter_forecasting=splitter_forecast)
    patient_data = initialized_dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="both",
        forecasting_override_variables_to_predict=["hemoglobin_-_718-7"],
    )

    last_date = patient_data["events"]["date"].max()

    # Forecasting should work
    assert f_split is not None
    assert f_split.sampled_variables == ["hemoglobin_-_718-7"]
    assert f_split.split_date_included_in_input == last_date
    assert f_split.split_date_included_in_input == pd.Timestamp("2016-05-13")
    assert f_split.lot_date == "override"

    # Inference: no target
    assert f_split.target_events_after_split.empty

    # Full patient history should be used as input
    assert f_split.events_until_split.shape == (78, 8)
    assert (f_split.events_until_split["date"] <= f_split.split_date_included_in_input).all()

    # Constant data integrity
    assert f_split.constant_data["patientid"].iloc[0] == "p0"
    assert f_split.constant_data.shape[0] == 1

    # Events should be None because no events splitter is set
    assert e_split is None


def test_inference_both_type_with_only_events(initialized_dm, mock_config):
    """Test that inference_type='both' gracefully returns None for the missing splitter."""
    splitter_events = DataSplitterEvents(
        initialized_dm,
        config=mock_config,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
    )
    splitter_events.setup_variables()

    data_splitter = DataSplitter(data_splitter_events=splitter_events)
    patient_data = initialized_dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="both",
        events_override_category="death",
        events_override_observation_time_delta=pd.Timedelta(weeks=52),
    )

    last_date = patient_data["events"]["date"].max()

    # Forecasting should be None because no forecasting splitter is set
    assert f_split is None

    # Events should work
    assert e_split is not None
    assert e_split.split_date_included_in_input == last_date
    assert e_split.split_date_included_in_input == pd.Timestamp("2016-05-13")
    assert e_split.sampled_category == "death"
    assert e_split.sampled_category_name == "death"

    # Observation window and outcome
    assert e_split.event_occurred is False
    assert e_split.event_censored == "end_of_data"
    assert e_split.observation_end_date == pd.Timestamp("2017-05-12")

    # Full patient history should be used as input
    assert e_split.events_until_split.shape == (78, 8)
    assert (e_split.events_until_split["date"] <= e_split.split_date_included_in_input).all()

    # Constant data integrity
    assert e_split.constant_data["patientid"].iloc[0] == "p0"
    assert e_split.constant_data.shape[0] == 1


# ────────────────────────────────────────────────────────────────────────────
# Test that forecasting truncation uses split_event_category (not just LoT)
# ────────────────────────────────────────────────────────────────────────────


def test_forecasting_truncates_at_next_split_event_not_just_lot():
    """
    Verify that _generate_variable_splits_for_date truncates target events
    at the next *split event* (config.split_event_category).

    Scenario (split_event_category = "custom_split"):
      Timeline for a single patient:
        Day 0   - custom_split event  (the split event that anchors the window)
        Day 5   - lab measurement     (before split date, input)
        Day 10  - split date          (curr_date)
        Day 15  - lab measurement     (target - should be kept)
        Day 20  - next custom_split   (next split event - target boundary)
        Day 25  - lab measurement     (target - should be EXCLUDED)
        Day 30  - lot event           (LoT - should NOT be the boundary)
        Day 35  - lab measurement     (target - should be EXCLUDED)

    With the old code, the target would
    include days 15, 25, and 35 (cutting only at day 30 LoT).
    With the fix (filtering by split_event_category), the target should
    include only day 15 (cutting at day 20 custom_split).
    """
    from twinweaver.common.config import Config

    cfg = Config()
    cfg.split_event_category = "custom_split"
    cfg.event_category_forecast = ["lab"]

    base_date = pd.Timestamp("2020-01-01")

    events = pd.DataFrame(
        {
            cfg.date_col: [base_date + pd.Timedelta(days=d) for d in [0, 5, 10, 15, 20, 25, 30, 35]],
            cfg.event_category_col: [
                "custom_split",  # Day 0: split event
                "lab",  # Day 5: lab (input)
                "lab",  # Day 10: lab at split date (input)
                "lab",  # Day 15: lab (target, before next split event)
                "custom_split",  # Day 20: next split event
                "lab",  # Day 25: lab (target, after next split event - exclude)
                "lot",  # Day 30: lot event (should NOT be the boundary)
                "lab",  # Day 35: lab (target, after lot - exclude)
            ],
            cfg.event_name_col: [
                "split_marker",
                "hemoglobin",
                "hemoglobin",
                "hemoglobin",
                "split_marker",
                "hemoglobin",
                "lot_marker",
                "hemoglobin",
            ],
            cfg.event_value_col: [
                "start",
                "13.0",
                "13.1",
                "13.2",
                "start",
                "13.3",
                "LoT Start",
                "13.4",
            ],
            cfg.event_descriptive_name_col: [
                "split marker",
                "hemoglobin",
                "hemoglobin",
                "hemoglobin",
                "split marker",
                "hemoglobin",
                "LoT",
                "hemoglobin",
            ],
            cfg.source_col: ["events"] * 8,
            cfg.meta_data_col: [pd.NA] * 8,
        }
    )

    constant = pd.DataFrame(
        {
            cfg.patient_id_col: ["p_test"],
            cfg.constant_split_col: ["train"],
        }
    )

    patient_data = {"events": events, "constant": constant}
    curr_date = base_date + pd.Timedelta(days=10)
    lot_date = base_date  # The split event that anchors this window

    # Build a minimal all_possible_split_dates with hemoglobin valid at curr_date
    all_possible_split_dates = pd.DataFrame(
        {
            cfg.date_col: [curr_date],
            cfg.event_name_col: ["hemoglobin"],
            cfg.event_category_col: ["lab"],
            "lot_date": [lot_date],
        }
    )

    # Create a DataManager stub - we only need dm.variable_types for the splitter
    dm = DataManager.__new__(DataManager)
    dm.config = cfg
    dm.variable_types = {"hemoglobin": "numeric"}
    dm.data_frames = {}
    dm.all_patientids = ["p_test"]

    splitter = DataSplitterForecasting(
        config=cfg,
        data_manager=dm,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_lookback_time_for_value=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        sampling_strategy="uniform",
    )

    np.random.seed(42)

    (date_splits, valid_sample_date, date_splits_meta, _) = splitter._generate_variable_splits_for_date(
        curr_date=curr_date,
        nr_samples=1,
        override_variables_to_predict=["hemoglobin"],
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=lot_date,
    )

    assert valid_sample_date is True
    assert len(date_splits) == 1

    target = date_splits[0].target_events_after_split

    # The target must only include lab events BEFORE the next custom_split (day 20).
    # So only the measurement at day 15 should survive.
    assert target.shape[0] == 1, (
        f"Expected 1 target event (day 15 only), got {target.shape[0]}. "
        f"Dates in target: {target[cfg.date_col].tolist()}"
    )
    assert target[cfg.date_col].iloc[0] == base_date + pd.Timedelta(days=15)

    # Also verify input events include everything up to and including curr_date
    input_events = date_splits[0].events_until_split
    assert (input_events[cfg.date_col] <= curr_date).all()
    assert input_events.shape[0] == 3  # Day 0, 5, 10


def test_forecasting_truncation_allow_beyond_next_split_date():
    """
    Verify that when allow_forecasting_beyond_next_split_date=True,
    target events are NOT truncated at the next split event.
    """
    from twinweaver.common.config import Config

    cfg = Config()
    cfg.split_event_category = "custom_split"
    cfg.event_category_forecast = ["lab"]

    base_date = pd.Timestamp("2020-01-01")

    events = pd.DataFrame(
        {
            cfg.date_col: [base_date + pd.Timedelta(days=d) for d in [0, 5, 10, 15, 20, 25]],
            cfg.event_category_col: [
                "custom_split",
                "lab",
                "lab",
                "lab",
                "custom_split",
                "lab",
            ],
            cfg.event_name_col: [
                "split_marker",
                "hemoglobin",
                "hemoglobin",
                "hemoglobin",
                "split_marker",
                "hemoglobin",
            ],
            cfg.event_value_col: ["start", "13.0", "13.1", "13.2", "start", "13.3"],
            cfg.event_descriptive_name_col: [
                "split marker",
                "hemoglobin",
                "hemoglobin",
                "hemoglobin",
                "split marker",
                "hemoglobin",
            ],
            cfg.source_col: ["events"] * 6,
            cfg.meta_data_col: [pd.NA] * 6,
        }
    )

    constant = pd.DataFrame(
        {
            cfg.patient_id_col: ["p_test"],
            cfg.constant_split_col: ["train"],
        }
    )

    patient_data = {"events": events, "constant": constant}
    curr_date = base_date + pd.Timedelta(days=10)
    lot_date = base_date

    all_possible_split_dates = pd.DataFrame(
        {
            cfg.date_col: [curr_date],
            cfg.event_name_col: ["hemoglobin"],
            cfg.event_category_col: ["lab"],
            "lot_date": [lot_date],
        }
    )

    dm = DataManager.__new__(DataManager)
    dm.config = cfg
    dm.variable_types = {"hemoglobin": "numeric"}
    dm.data_frames = {}
    dm.all_patientids = ["p_test"]

    splitter = DataSplitterForecasting(
        config=cfg,
        data_manager=dm,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_lookback_time_for_value=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        sampling_strategy="uniform",
        allow_forecasting_beyond_next_split_date=True,
    )

    np.random.seed(42)

    (date_splits, valid_sample_date, _, _) = splitter._generate_variable_splits_for_date(
        curr_date=curr_date,
        nr_samples=1,
        override_variables_to_predict=["hemoglobin"],
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=lot_date,
    )

    assert valid_sample_date is True
    assert len(date_splits) == 1

    target = date_splits[0].target_events_after_split

    # With allow_forecasting_beyond_next_split_date=True, no truncation at
    # the next split event, so both day 15 and day 25 labs should be in target.
    assert target.shape[0] == 2, (
        f"Expected 2 target events (days 15 and 25), got {target.shape[0]}. "
        f"Dates in target: {target[cfg.date_col].tolist()}"
    )
    expected_dates = [
        base_date + pd.Timedelta(days=15),
        base_date + pd.Timedelta(days=25),
    ]
    assert target[cfg.date_col].tolist() == expected_dates


# ────────────────────────────────────────────────────────────────────────────
# Test that DataSplitterEvents respects unit_length_to_sample
# ────────────────────────────────────────────────────────────────────────────


def test_events_splitter_respects_unit_length_to_sample():
    """
    Verify that the observation_end_date produced by DataSplitterEvents is
    bounded by ``max_length_to_sample`` expressed in the correct
    ``unit_length_to_sample``, and is NOT pushed forward to the next event
    when that event lies beyond the sampled prediction window.

    Scenario (unit_length_to_sample = "days", max = 7 days, min = 1 day):
      Timeline for patient p_test:
        Day 0   - lot event            (split event)
        Day 0   - lab measurement      (input – at the split date)
        Day 100 - death event          (far in the future)

      We split at Day 0 and predict "death" with a window of at most 7 days.
      The observation_end_date must be <= Day 0 + 7 days.  Before the fix
      it would jump to Day 100 (the next event), violating the window.
    """
    from twinweaver.common.config import Config

    cfg = Config()
    cfg.seed = 42
    cfg.split_event_category = "lot"
    cfg.event_category_events_prediction_with_naming = {"death": "death"}
    cfg.constant_columns_to_use = []

    base_date = pd.Timestamp("2020-01-01")

    events = pd.DataFrame(
        {
            cfg.date_col: [
                base_date,
                base_date,
                base_date + pd.Timedelta(days=100),
            ],
            cfg.event_category_col: ["lot", "lab", "death"],
            cfg.event_name_col: ["line_number", "hemoglobin", "death"],
            cfg.event_value_col: ["1", "13.0", "deceased"],
            cfg.event_descriptive_name_col: ["line number", "hemoglobin", "death"],
            cfg.source_col: ["events"] * 3,
            cfg.meta_data_col: [pd.NA] * 3,
        }
    )

    constant = pd.DataFrame(
        {
            cfg.patient_id_col: ["p_test"],
            cfg.constant_split_col: ["train"],
        }
    )

    patient_data = {"events": events, "constant": constant}

    # Create a minimal DataManager stub
    dm = DataManager.__new__(DataManager)
    dm.config = cfg
    dm.data_frames = {"events": events}
    dm.all_patientids = ["p_test"]

    max_length = pd.Timedelta(days=7)
    min_length = pd.Timedelta(days=1)

    splitter = DataSplitterEvents(
        data_manager=dm,
        config=cfg,
        max_length_to_sample=max_length,
        min_length_to_sample=min_length,
        unit_length_to_sample="days",
        max_split_length_after_split_event=pd.Timedelta(days=0),
    )
    splitter.setup_variables()

    np.random.seed(cfg.seed)

    # Use override split dates and category to make the test deterministic
    splits = splitter.get_splits_from_patient(
        patient_data,
        max_nr_samples_per_split=1,
        override_split_dates=[base_date],
        override_category="death",
    )

    assert len(splits) == 1
    assert len(splits[0]) == 1

    option = splits[0][0]

    # The critical assertion: observation_end_date must respect the sampled
    # window which is at most base_date + 7 days.  Before the fix it was
    # pushed to base_date + 100 days (the death event).
    assert option.observation_end_date <= base_date + max_length, (
        f"observation_end_date ({option.observation_end_date}) exceeded the "
        f"maximum prediction window ({base_date + max_length}).  "
        f"unit_length_to_sample is not being respected."
    )

    # The event (death) is at Day 100, well outside the 7-day window,
    # so it should NOT have occurred.
    assert option.event_occurred is False

    # The end_date is within the data range (data goes to Day 100), so the
    # event simply did not occur within the prediction window — not censored.
    assert option.event_censored is None


def test_events_splitter_unit_days_vs_weeks():
    """
    Verify that changing ``unit_length_to_sample`` between 'days' and 'weeks'
    actually produces different observation windows when
    ``max_length_to_sample`` is the same Timedelta.

    With max_length_to_sample = 14 days:
      - unit='days'  → random window in [1 … 14] days
      - unit='weeks' → random window in [1 … 2]  weeks (= 7 or 14 days)

    By running many samples we can verify the units produce the expected
    granularity.
    """
    from twinweaver.common.config import Config

    cfg = Config()
    cfg.seed = 123
    cfg.split_event_category = "lot"
    cfg.event_category_events_prediction_with_naming = {"death": "death"}
    cfg.constant_columns_to_use = []

    base_date = pd.Timestamp("2020-01-01")
    far_future = base_date + pd.Timedelta(days=365)

    events = pd.DataFrame(
        {
            cfg.date_col: [base_date, base_date, far_future],
            cfg.event_category_col: ["lot", "lab", "death"],
            cfg.event_name_col: ["line_number", "hemoglobin", "death"],
            cfg.event_value_col: ["1", "13.0", "deceased"],
            cfg.event_descriptive_name_col: ["line number", "hemoglobin", "death"],
            cfg.source_col: ["events"] * 3,
            cfg.meta_data_col: [pd.NA] * 3,
        }
    )

    constant = pd.DataFrame(
        {
            cfg.patient_id_col: ["p_test"],
            cfg.constant_split_col: ["train"],
        }
    )
    patient_data = {"events": events, "constant": constant}

    dm = DataManager.__new__(DataManager)
    dm.config = cfg
    dm.data_frames = {"events": events}
    dm.all_patientids = ["p_test"]

    max_td = pd.Timedelta(days=14)
    min_td = pd.Timedelta(days=1)

    # --- unit = "days" ---
    splitter_days = DataSplitterEvents(
        data_manager=dm,
        config=cfg,
        max_length_to_sample=max_td,
        min_length_to_sample=min_td,
        unit_length_to_sample="days",
        max_split_length_after_split_event=pd.Timedelta(days=0),
    )
    splitter_days.setup_variables()

    day_deltas = set()
    for i in range(200):
        np.random.seed(i)
        splits = splitter_days.get_splits_from_patient(
            patient_data,
            max_nr_samples_per_split=1,
            override_split_dates=[base_date],
            override_category="death",
        )
        delta = (splits[0][0].observation_end_date - base_date).days
        day_deltas.add(delta)

    # With unit="days" and range [1..14], we expect many distinct day values
    assert len(day_deltas) > 2, f"With unit='days', expected many distinct day offsets but got {day_deltas}"
    # All values should be within [1, 14]
    assert min(day_deltas) >= 1
    assert max(day_deltas) <= 14

    # --- unit = "weeks" ---
    splitter_weeks = DataSplitterEvents(
        data_manager=dm,
        config=cfg,
        max_length_to_sample=max_td,
        min_length_to_sample=min_td,
        unit_length_to_sample="weeks",
        max_split_length_after_split_event=pd.Timedelta(days=0),
    )
    splitter_weeks.setup_variables()

    week_deltas = set()
    for i in range(200):
        np.random.seed(i)
        splits = splitter_weeks.get_splits_from_patient(
            patient_data,
            max_nr_samples_per_split=1,
            override_split_dates=[base_date],
            override_category="death",
        )
        delta = (splits[0][0].observation_end_date - base_date).days
        week_deltas.add(delta)

    # With unit="weeks", min_units=0 (1 day // 7 = 0), max_units=2 (14 days // 7 = 2)
    # So we get 0, 1, or 2 weeks → 0, 7, or 14 days
    # All offsets must be multiples of 7
    for d in week_deltas:
        assert d % 7 == 0, (
            f"With unit='weeks', got a non-week-multiple offset of {d} days. "
            f"unit_length_to_sample is not being respected."
        )


# ────────────────────────────────────────────────────────────────────────────
# Helpers for the synthetic-timeline tests below
# ────────────────────────────────────────────────────────────────────────────

_BASE_DATE = pd.Timestamp("2020-01-01")


def _make_events(cfg, rows):
    """
    Build a synthetic events DataFrame from compact row tuples.

    Parameters
    ----------
    cfg : Config
        Config providing the column names.
    rows : list[tuple]
        Each row is ``(day_offset, event_category, event_name, event_value)``. The descriptive name
        is derived from the event name by replacing underscores with spaces.

    Returns
    -------
    pd.DataFrame
        Events for a single synthetic patient, dated relative to ``_BASE_DATE``.
    """

    return pd.DataFrame(
        {
            cfg.date_col: [_BASE_DATE + pd.Timedelta(days=row[0]) for row in rows],
            cfg.event_category_col: [row[1] for row in rows],
            cfg.event_name_col: [row[2] for row in rows],
            cfg.event_value_col: [row[3] for row in rows],
            cfg.event_descriptive_name_col: [row[2].replace("_", " ") for row in rows],
            cfg.source_col: ["events"] * len(rows),
            cfg.meta_data_col: [pd.NA] * len(rows),
        }
    )


def _make_constant(cfg):
    """Build the minimal constant DataFrame for a single synthetic patient."""

    return pd.DataFrame({cfg.patient_id_col: ["p_test"], cfg.constant_split_col: ["train"]})


def _make_dm_stub(cfg, variable_types):
    """Create a DataManager stub - the splitters only need config, variable_types and patient ids."""

    dm = DataManager.__new__(DataManager)
    dm.config = cfg
    dm.variable_types = variable_types
    dm.data_frames = {}
    dm.all_patientids = ["p_test"]
    return dm


def _base_config(split_event_category="lot", forecast_categories=None):
    """Create a fresh Config wired for the synthetic timelines used below."""

    from twinweaver.common.config import Config

    cfg = Config()
    cfg.split_event_category = split_event_category
    cfg.event_category_forecast = ["lab"] if forecast_categories is None else forecast_categories
    return cfg


def _days(dates):
    """Convert a series/list of dates into integer day offsets relative to ``_BASE_DATE``."""

    return sorted(int((pd.Timestamp(date) - _BASE_DATE).days) for date in dates)


# ────────────────────────────────────────────────────────────────────────────
# select_random_splits: seeding and sampling without replacement
# ────────────────────────────────────────────────────────────────────────────


def _make_candidate_dates(cfg, per_split_event):
    """
    Build a candidate split-date frame as returned by _get_all_dates_within_range_of_split_event.

    Parameters
    ----------
    cfg : Config
        Config providing the column names.
    per_split_event : dict[int, list[int]]
        Maps the day offset of each split event to the day offsets of its candidate split dates.
    """

    rows = []
    for split_day, candidate_days in per_split_event.items():
        for candidate_day in candidate_days:
            rows.append(
                {
                    cfg.date_col: _BASE_DATE + pd.Timedelta(days=candidate_day),
                    cfg.split_date_col: _BASE_DATE + pd.Timedelta(days=split_day),
                }
            )
    return pd.DataFrame(rows)


def _make_base_splitter(cfg, random_state=None):
    """Create a bare BaseDataSplitter for testing select_random_splits in isolation."""

    return BaseDataSplitter(
        data_manager=_make_dm_stub(cfg, {}),
        config=cfg,
        max_split_length_after_split_event=pd.Timedelta(days=90),
        random_state=random_state,
    )


def test_select_random_splits_samples_without_replacement():
    """
    Verify that select_random_splits never returns the same split date twice.

    Previously the implementation used ``.sample(n=..., replace=True, random_state=1)``, so asking
    for 3 split dates could return the same date up to 3 times. Sampling without replacement must
    return 3 *distinct* dates out of the 5 candidates.
    """

    cfg = _base_config()
    splitter = _make_base_splitter(cfg)
    candidates = _make_candidate_dates(cfg, {0: [0, 5, 10, 15, 20]})

    for seed in range(25):
        np.random.seed(seed)
        selected = splitter.select_random_splits(candidates, max_num_splits_per_split_event=3)

        assert selected.shape[0] == 3, f"Expected 3 split dates for seed {seed}, got {selected.shape[0]}"
        assert selected[cfg.date_col].nunique() == 3, (
            f"Duplicate split dates returned for seed {seed}: {_days(selected[cfg.date_col])}"
        )
        assert set(_days(selected[cfg.date_col])).issubset({0, 5, 10, 15, 20})


def test_select_random_splits_caps_at_number_of_candidates():
    """
    Verify that requesting more splits than there are candidate dates returns all candidates.

    ``replace=True`` used to hide this case by padding the result with duplicates; capping is the
    correct behaviour and must not raise.
    """

    cfg = _base_config()
    splitter = _make_base_splitter(cfg)
    candidates = _make_candidate_dates(cfg, {0: [0, 5]})

    np.random.seed(0)
    selected = splitter.select_random_splits(candidates, max_num_splits_per_split_event=5)

    assert selected.shape[0] == 2
    assert _days(selected[cfg.date_col]) == [0, 5]


def test_select_random_splits_caps_per_split_event():
    """Verify that the cap is applied per split event, not across the whole frame."""

    cfg = _base_config()
    splitter = _make_base_splitter(cfg)
    candidates = _make_candidate_dates(cfg, {0: [0, 1, 2, 3], 100: [100], 200: [200, 201, 202]})

    np.random.seed(7)
    selected = splitter.select_random_splits(candidates, max_num_splits_per_split_event=2)

    counts = selected.groupby(cfg.split_date_col).size().to_dict()
    assert sorted(counts.values()) == [1, 2, 2]  # split event at day 100 only has one candidate
    assert selected.shape[0] == 5
    assert selected[cfg.date_col].nunique() == 5


def test_select_random_splits_respects_config_seed():
    """
    Verify that split-date selection depends on Config.seed.

    ``random_state=1`` was hardcoded, so every patient, every run and every process drew the exact
    same split dates and changing ``Config.seed`` had no effect at all.
    """

    cfg = _base_config()
    splitter = _make_base_splitter(cfg)
    candidates = _make_candidate_dates(cfg, {0: list(range(20))})

    selections_per_seed = {}
    for seed in [1, 2, 3, 4, 5]:
        cfg.seed = seed  # re-seeds the global NumPy stream
        selected = splitter.select_random_splits(candidates, max_num_splits_per_split_event=3)
        selections_per_seed[seed] = tuple(_days(selected[cfg.date_col]))

    assert len(set(selections_per_seed.values())) > 1, (
        f"Split-date selection did not change with Config.seed: {selections_per_seed}"
    )

    #: the same seed must still reproduce the same selection
    cfg.seed = 1
    repeated = splitter.select_random_splits(candidates, max_num_splits_per_split_event=3)
    assert tuple(_days(repeated[cfg.date_col])) == selections_per_seed[1]


def test_select_random_splits_advances_between_calls():
    """
    Verify that consecutive calls do not all draw identically.

    With the hardcoded ``random_state=1`` every call returned the same dates, so all patients got
    the same relative split position.
    """

    cfg = _base_config()
    cfg.seed = 42
    splitter = _make_base_splitter(cfg)
    candidates = _make_candidate_dates(cfg, {0: list(range(20))})

    selections = [
        tuple(_days(splitter.select_random_splits(candidates, max_num_splits_per_split_event=2)[cfg.date_col]))
        for _ in range(10)
    ]

    assert len(set(selections)) > 1, f"All consecutive calls returned the same split dates: {selections}"


def test_select_random_splits_explicit_random_state():
    """Verify that an explicit random_state makes the selection independent of the global stream."""

    cfg = _base_config()
    candidates = _make_candidate_dates(cfg, {0: list(range(20))})

    np.random.seed(0)
    first = _make_base_splitter(cfg, random_state=123).select_random_splits(
        candidates, max_num_splits_per_split_event=3
    )
    np.random.seed(999)  # different global state must not matter
    second = _make_base_splitter(cfg, random_state=123).select_random_splits(
        candidates, max_num_splits_per_split_event=3
    )

    assert _days(first[cfg.date_col]) == _days(second[cfg.date_col])

    different = _make_base_splitter(cfg, random_state=456).select_random_splits(
        candidates, max_num_splits_per_split_event=3
    )
    assert _days(different[cfg.date_col]) != _days(first[cfg.date_col])


def test_select_random_splits_empty_input():
    """Verify that an empty candidate frame is passed through instead of raising."""

    cfg = _base_config()
    splitter = _make_base_splitter(cfg)
    empty = pd.DataFrame(columns=[cfg.date_col, cfg.split_date_col])

    selected = splitter.select_random_splits(empty, max_num_splits_per_split_event=3)
    assert selected.shape[0] == 0


# ────────────────────────────────────────────────────────────────────────────
# min_total_horizon: require the split to cover a minimum forecast horizon
# ────────────────────────────────────────────────────────────────────────────


def _make_horizon_splitter(cfg, dm, min_total_horizon, **kwargs):
    """Create a forecasting splitter for the horizon tests."""

    return DataSplitterForecasting(
        config=cfg,
        data_manager=dm,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_lookback_time_for_value=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=0),
        min_total_horizon=min_total_horizon,
        sampling_strategy="uniform",
        **kwargs,
    )


def test_min_total_horizon_rejects_splits_that_are_too_short():
    """
    Verify that a variable is not eligible when none of its future values reaches min_total_horizon.

    Scenario (split_event_category = "lot", forecast horizon 90 days):
        Day 0   - lot event (anchors the only candidate split date)
        Day 0   - hemoglobin (input, satisfies min_nr_variable_seen_previously)
        Day 10  - hemoglobin (target, only 10 days after the split date)
        Day 200 - progression (last date, excluded from split dates)

    With min_total_horizon = 30 days the split covers only 10 days, so hemoglobin must not be
    offered as a forecasting variable at day 0. Without min_total_horizon it must be offered.
    """

    cfg = _base_config()
    events = _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (0, "lab", "hemoglobin", "13.0"),
            (10, "lab", "hemoglobin", "13.1"),
            (200, "progression", "progression_marker", "progressed"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})

    #: without a minimum horizon the 10-day target is accepted
    splits_no_horizon, _ = _make_horizon_splitter(cfg, dm, None)._get_all_possible_splits(
        patient_data, list_of_valid_categories=["lab"]
    )
    valid_no_horizon = splits_no_horizon[splits_no_horizon["event_name"].notna()]
    assert valid_no_horizon.shape[0] == 1
    assert valid_no_horizon["event_name"].iloc[0] == "hemoglobin"

    #: with a 30-day minimum horizon it is rejected
    splits_with_horizon, _ = _make_horizon_splitter(cfg, dm, pd.Timedelta(days=30))._get_all_possible_splits(
        patient_data, list_of_valid_categories=["lab"]
    )
    valid_with_horizon = splits_with_horizon[splits_with_horizon["event_name"].notna()]
    assert valid_with_horizon.shape[0] == 0, (
        f"Expected no eligible variable, got {valid_with_horizon['event_name'].tolist()}"
    )


def test_min_total_horizon_accepts_splits_that_reach_the_horizon():
    """
    Verify that a variable stays eligible as soon as one future value reaches min_total_horizon.

    Same timeline as the rejection test, plus a hemoglobin measurement at day 60. The day-60 value
    is at least 30 days after the split date, so the variable is eligible and the resulting target
    keeps both future measurements.
    """

    cfg = _base_config()
    events = _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (0, "lab", "hemoglobin", "13.0"),
            (10, "lab", "hemoglobin", "13.1"),
            (60, "lab", "hemoglobin", "13.2"),
            (200, "progression", "progression_marker", "progressed"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})
    splitter = _make_horizon_splitter(cfg, dm, pd.Timedelta(days=30))

    all_possible_split_dates, _ = splitter._get_all_possible_splits(patient_data, list_of_valid_categories=["lab"])
    valid = all_possible_split_dates[all_possible_split_dates["event_name"].notna()]
    assert valid.shape[0] == 1
    assert valid["event_name"].iloc[0] == "hemoglobin"

    np.random.seed(42)
    date_splits, valid_date, _, _ = splitter._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=1,
        override_variables_to_predict=["hemoglobin"],
        events=events,
        all_possible_split_dates=valid.rename(columns={"event_name": cfg.event_name_col}),
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )

    assert valid_date is True
    assert len(date_splits) == 1
    assert _days(date_splits[0].target_events_after_split[cfg.date_col]) == [10, 60]


def test_min_total_horizon_rechecked_after_truncation_at_next_split_event():
    """
    Verify that the horizon is re-checked on the *final* target, not only during eligibility.

    Scenario (split_event_category = "custom_split", forecast horizon 90 days):
        Day 0  - custom_split (anchors the split date)
        Day 0  - hemoglobin (input)
        Day 10 - hemoglobin (target, only 10 days out)
        Day 20 - custom_split (next split event -> target is truncated here)
        Day 40 - hemoglobin (would satisfy a 30-day horizon, but is cut off)
        Day 60 - progression (last date)

    Eligibility sees the day-40 value and accepts hemoglobin, but truncation at the next split event
    leaves only the day-10 value. The generated split therefore covers 10 days and must be
    discarded when min_total_horizon is 30 days.
    """

    cfg = _base_config(split_event_category="custom_split")
    events = _make_events(
        cfg,
        [
            (0, "custom_split", "split_marker", "start"),
            (0, "lab", "hemoglobin", "13.0"),
            (10, "lab", "hemoglobin", "13.1"),
            (20, "custom_split", "split_marker", "start"),
            (40, "lab", "hemoglobin", "13.2"),
            (60, "progression", "progression_marker", "progressed"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})
    all_possible_split_dates = pd.DataFrame(
        {
            cfg.date_col: [_BASE_DATE],
            cfg.event_name_col: ["hemoglobin"],
            cfg.event_category_col: ["lab"],
            "lot_date": [_BASE_DATE],
        }
    )

    #: eligibility passes, because the raw future events contain the day-40 value
    splitter_with_horizon = _make_horizon_splitter(cfg, dm, pd.Timedelta(days=30))
    eligible, _ = splitter_with_horizon._get_all_possible_splits(patient_data, list_of_valid_categories=["lab"])
    assert eligible[eligible["event_name"].notna()].shape[0] == 1

    #: but the truncated target only reaches day 10, so the split is rejected
    np.random.seed(42)
    date_splits, valid_date, _, _ = splitter_with_horizon._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=1,
        override_variables_to_predict=["hemoglobin"],
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )
    assert valid_date is True
    assert len(date_splits) == 0, (
        "Expected the split to be rejected because truncation at the next split event left only a "
        f"10-day horizon, got {len(date_splits)} split(s)."
    )

    #: without a minimum horizon the same (short) split is produced
    np.random.seed(42)
    date_splits_no_horizon, _, _, _ = _make_horizon_splitter(cfg, dm, None)._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=1,
        override_variables_to_predict=["hemoglobin"],
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )
    assert len(date_splits_no_horizon) == 1
    assert _days(date_splits_no_horizon[0].target_events_after_split[cfg.date_col]) == [10]


def test_min_total_horizon_not_applied_for_override_split_dates():
    """
    Verify that inference splits are not rejected by min_total_horizon.

    During inference there is no target at all, so the horizon check must be skipped exactly like
    the existing "empty target" check is skipped when override_split_dates is given.
    """

    cfg = _base_config()
    events = _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (0, "lab", "hemoglobin", "13.0"),
            (10, "lab", "hemoglobin", "13.1"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})
    splitter = _make_horizon_splitter(cfg, dm, pd.Timedelta(days=30))

    all_possible_split_dates = pd.DataFrame(
        {
            cfg.date_col: [_BASE_DATE],
            cfg.event_name_col: ["hemoglobin"],
            cfg.split_date_col: ["override"],
        }
    )

    np.random.seed(42)
    date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=1,
        override_variables_to_predict=["hemoglobin"],
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=[_BASE_DATE],
        patient_data=patient_data,
        lot_date=None,
    )

    assert len(date_splits) == 1


@pytest.mark.parametrize(
    "min_total_horizon",
    [pd.Timedelta(days=0), pd.Timedelta(days=-1), pd.Timedelta(days=91), 30],
)
def test_min_total_horizon_validation(min_total_horizon):
    """Verify that invalid min_total_horizon values are rejected at construction time."""

    cfg = _base_config()
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})

    with pytest.raises(ValueError):
        DataSplitterForecasting(
            config=cfg,
            data_manager=dm,
            max_forecasted_trajectory_length=pd.Timedelta(days=90),
            min_total_horizon=min_total_horizon,
        )


def test_min_total_horizon_end_to_end_on_test_data(initialized_dm, mock_config):
    """Verify that every generated target reaches min_total_horizon for the committed test data."""

    min_total_horizon = pd.Timedelta(days=30)
    splitter_forecast = DataSplitterForecasting(
        data_manager=initialized_dm,
        config=mock_config,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        min_total_horizon=min_total_horizon,
    )
    splitter_forecast.setup_statistics()

    nr_checked = 0
    for patientid in initialized_dm.all_patientids:
        splits = splitter_forecast.get_splits_from_patient(
            initialized_dm.get_patient_data(patientid), nr_samples_per_split=1
        )
        for group in splits:
            if group is None:
                continue
            for option in group:
                covered = option.target_events_after_split["date"].max() - option.split_date_included_in_input
                assert covered >= min_total_horizon, (
                    f"Patient {patientid} split at {option.split_date_included_in_input} only covers {covered}"
                )
                nr_checked += 1

    assert nr_checked > 0, "No splits were generated, so the horizon requirement was not exercised"


# ────────────────────────────────────────────────────────────────────────────
# no_split_before_events: gate an index event before which no split may happen
# ────────────────────────────────────────────────────────────────────────────


def _gated_timeline(cfg):
    """
    Build the timeline shared by the gating tests.

        Day 0  - lot event      (split event *before* the gate event)
        Day 5  - hemoglobin
        Day 10 - treatment start (the gate event: category "treatment", name "tx_start")
        Day 10 - lot event      (split event on the gate date)
        Day 20 - hemoglobin
        Day 40 - hemoglobin     (last date, always excluded from split dates)
    """

    return _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (5, "lab", "hemoglobin", "13.0"),
            (10, "treatment", "tx_start", "started"),
            (10, "lot", "lot_marker", "LoT Start"),
            (20, "lab", "hemoglobin", "13.1"),
            (40, "lab", "hemoglobin", "13.2"),
        ],
    )


@pytest.mark.parametrize("gate", [["treatment"], ["tx_start"], ["treatment", "something_else"]])
def test_no_split_before_events_removes_earlier_dates(gate):
    """
    Verify that no split date before the gate event survives.

    The gate is matched against both the event category ("treatment") and the event name
    ("tx_start"), so either spelling gates the same way. Days 0 and 5 lie before the gate event at
    day 10 and must disappear; day 10 (the gate date itself) and day 20 must remain.
    """

    cfg = _base_config()
    patient_data = {"events": _gated_timeline(cfg), "constant": _make_constant(cfg)}

    ungated = _make_base_splitter(cfg)._get_all_dates_within_range_of_split_event(
        patient_data, pd.Timedelta(0), pd.Timedelta(days=90)
    )
    assert _days(ungated[cfg.date_col]) == [0, 5, 10, 20]

    gated_splitter = BaseDataSplitter(
        data_manager=_make_dm_stub(cfg, {}),
        config=cfg,
        max_split_length_after_split_event=pd.Timedelta(days=90),
        no_split_before_events=gate,
    )
    gated = gated_splitter._get_all_dates_within_range_of_split_event(
        patient_data, pd.Timedelta(0), pd.Timedelta(days=90)
    )

    assert _days(gated[cfg.date_col]) == [10, 20], (
        f"Split dates before the gate event were not removed: {_days(gated[cfg.date_col])}"
    )
    #: the gate date itself is allowed - "no split *before* the event"
    assert 10 in _days(gated[cfg.date_col])


def test_no_split_before_events_without_gate_event_yields_no_splits():
    """
    Verify that a patient without any gate event produces no splits at all.

    This is the conservative behaviour: every split has to be anchored after the gate event, so a
    patient who never had it cannot contribute a split.
    """

    cfg = _base_config()
    events = _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (0, "lab", "hemoglobin", "13.0"),
            (30, "lab", "hemoglobin", "13.1"),
            (60, "lab", "hemoglobin", "13.2"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})

    #: base splitter: no candidate dates at all
    gated_base = BaseDataSplitter(
        data_manager=dm,
        config=cfg,
        max_split_length_after_split_event=pd.Timedelta(days=90),
        no_split_before_events=["treatment"],
    )
    candidates = gated_base._get_all_dates_within_range_of_split_event(
        patient_data, pd.Timedelta(0), pd.Timedelta(days=90)
    )
    assert candidates.shape[0] == 0
    assert list(candidates.columns) == [cfg.date_col, cfg.split_date_col]

    #: forecasting splitter: reports "no possible splits"
    splitter_forecast = DataSplitterForecasting(
        config=cfg,
        data_manager=dm,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        no_split_before_events=["treatment"],
        sampling_strategy="uniform",
    )
    assert splitter_forecast.get_splits_from_patient(patient_data, nr_samples_per_split=1) == [None]
    assert splitter_forecast.get_splits_from_patient(patient_data, nr_samples_per_split=1, include_metadata=True) == (
        [None],
        None,
    )

    #: events splitter: no splits either
    cfg.event_category_events_prediction_with_naming = {"death": "death"}
    splitter_events = DataSplitterEvents(
        dm,
        config=cfg,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        no_split_before_events=["treatment"],
    )
    splitter_events.manual_variables_category_mapping = {"death": "death"}
    assert splitter_events.get_splits_from_patient(patient_data, max_nr_samples_per_split=1) == []


def test_no_split_before_events_end_to_end_keeps_splitters_aligned():
    """
    Verify that gating works through the combined DataSplitter and keeps both splitters aligned.

    The forecasting splitter determines the split dates and hands them to the events splitter, so
    both must end up on the same, gated split date.
    """

    cfg = _base_config()
    cfg.event_category_events_prediction_with_naming = {"death": "death"}
    events = _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (0, "lab", "hemoglobin", "13.0"),
            (5, "lab", "hemoglobin", "13.1"),
            (10, "treatment", "tx_start", "started"),
            (10, "lot", "lot_marker", "LoT Start"),
            (20, "lab", "hemoglobin", "13.2"),
            (60, "lab", "hemoglobin", "13.3"),
            (120, "death", "death_marker", "death"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric"})

    splitter_forecast = DataSplitterForecasting(
        config=cfg,
        data_manager=dm,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        no_split_before_events=["treatment"],
        sampling_strategy="uniform",
    )
    splitter_events = DataSplitterEvents(
        dm,
        config=cfg,
        max_length_to_sample=pd.Timedelta(weeks=104),
        min_length_to_sample=pd.Timedelta(weeks=1),
        max_split_length_after_split_event=pd.Timedelta(days=90),
        no_split_before_events=["treatment"],
    )
    splitter_events.manual_variables_category_mapping = {"death": "death"}

    np.random.seed(42)
    forecasting_splits, events_splits, ref_dates = DataSplitter(
        splitter_events, splitter_forecast
    ).get_splits_from_patient_with_target(patient_data, max_num_splits_per_split_event=1)

    assert ref_dates is not None and ref_dates.shape[0] >= 1
    assert min(_days(ref_dates[cfg.date_col])) >= 10, (
        f"A split date before the gate event was used: {_days(ref_dates[cfg.date_col])}"
    )

    forecasting_dates = [option.split_date_included_in_input for group in forecasting_splits for option in group]
    events_dates = [option.split_date_included_in_input for group in events_splits for option in group]
    assert len(forecasting_dates) > 0
    assert set(forecasting_dates) == set(events_dates)
    assert all(day >= 10 for day in _days(forecasting_dates))


@pytest.mark.parametrize("gate", ["treatment", [], ["treatment", 5]])
def test_no_split_before_events_validation(gate):
    """Verify that a malformed no_split_before_events is rejected at construction time."""

    cfg = _base_config()
    with pytest.raises(AssertionError):
        BaseDataSplitter(
            data_manager=_make_dm_stub(cfg, {}),
            config=cfg,
            no_split_before_events=gate,
        )


# ────────────────────────────────────────────────────────────────────────────
# Multi-endpoint forecasting (e.g. ~15 endpoints per split)
# ────────────────────────────────────────────────────────────────────────────

_NR_ENDPOINTS = 15


def _multi_endpoint_setup(cfg, nr_endpoints=_NR_ENDPOINTS):
    """
    Build a timeline with `nr_endpoints` lab endpoints, each measured at day 0, day 30 and day 60.

        Day 0  - lot event + one measurement per endpoint (input)
        Day 30 - one measurement per endpoint (target)
        Day 60 - one measurement per endpoint (target)
        Day 200 - progression (last date, excluded from split dates)
    """

    endpoints = [f"endpoint_{idx:02d}" for idx in range(nr_endpoints)]
    rows = [(0, "lot", "lot_marker", "LoT Start")]
    for day in [0, 30, 60]:
        for idx, endpoint in enumerate(endpoints):
            rows.append((day, "lab", endpoint, str(10.0 + idx + day / 100.0)))
    rows.append((200, "progression", "progression_marker", "progressed"))

    events = _make_events(cfg, rows)
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {endpoint: "numeric" for endpoint in endpoints})
    all_possible_split_dates = pd.DataFrame(
        {
            cfg.date_col: [_BASE_DATE] * nr_endpoints,
            cfg.event_name_col: endpoints,
            cfg.event_category_col: ["lab"] * nr_endpoints,
            "lot_date": [_BASE_DATE] * nr_endpoints,
        }
    )
    return endpoints, events, patient_data, dm, all_possible_split_dates


def _multi_endpoint_splitter(cfg, dm, min_nr, max_nr, **kwargs):
    """Create a forecasting splitter configured for multi-endpoint sampling."""

    return DataSplitterForecasting(
        config=cfg,
        data_manager=dm,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_lookback_time_for_value=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=0),
        min_nr_variables_to_sample=min_nr,
        max_nr_variables_to_sample=max_nr,
        sampling_strategy="uniform",
        **kwargs,
    )


def test_forecasting_samples_all_requested_endpoints():
    """Verify that a split can carry all 15 endpoints when min and max are both set to 15."""

    cfg = _base_config()
    endpoints, events, patient_data, dm, all_possible_split_dates = _multi_endpoint_setup(cfg)
    splitter = _multi_endpoint_splitter(cfg, dm, _NR_ENDPOINTS, _NR_ENDPOINTS)

    np.random.seed(42)
    date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=1,
        override_variables_to_predict=None,
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )

    assert len(date_splits) == 1
    sampled = list(date_splits[0].sampled_variables)
    assert len(sampled) == _NR_ENDPOINTS, f"Expected {_NR_ENDPOINTS} endpoints, got {len(sampled)}: {sampled}"
    assert set(sampled) == set(endpoints)

    #: every endpoint must be present in the target, at both future dates
    target = date_splits[0].target_events_after_split
    assert set(target[cfg.event_name_col].unique()) == set(endpoints)
    assert target.shape[0] == _NR_ENDPOINTS * 2
    assert _days(target[cfg.date_col].unique()) == [30, 60]


def test_forecasting_eligibility_finds_all_endpoints():
    """Verify that _get_all_possible_splits offers all 15 endpoints at the candidate split date."""

    cfg = _base_config()
    endpoints, _, patient_data, dm, _ = _multi_endpoint_setup(cfg)
    splitter = _multi_endpoint_splitter(cfg, dm, _NR_ENDPOINTS, _NR_ENDPOINTS, min_total_horizon=pd.Timedelta(days=30))

    all_possible_split_dates, _ = splitter._get_all_possible_splits(patient_data, list_of_valid_categories=["lab"])
    valid = all_possible_split_dates[all_possible_split_dates["event_name"].notna()]

    assert set(valid["event_name"].tolist()) == set(endpoints)
    assert valid.shape[0] == _NR_ENDPOINTS


def test_forecasting_max_nr_variables_to_sample_is_reachable():
    """
    Verify that max_nr_variables_to_sample can actually be drawn.

    ``np.random.randint`` has an exclusive upper bound, so ``randint(min, max)`` could never return
    ``max`` - asking for up to 15 endpoints yielded at most 14. The bound is now inclusive.
    """

    cfg = _base_config()
    _, events, patient_data, dm, all_possible_split_dates = _multi_endpoint_setup(cfg)
    splitter = _multi_endpoint_splitter(cfg, dm, 1, _NR_ENDPOINTS)

    observed_counts = set()
    for seed in range(100):
        np.random.seed(seed)
        date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
            curr_date=_BASE_DATE,
            nr_samples=1,
            override_variables_to_predict=None,
            events=events,
            all_possible_split_dates=all_possible_split_dates,
            apply_filtering=False,
            override_split_dates=None,
            patient_data=patient_data,
            lot_date=_BASE_DATE,
        )
        if len(date_splits) == 1:
            observed_counts.add(len(date_splits[0].sampled_variables))

    assert _NR_ENDPOINTS in observed_counts, (
        f"max_nr_variables_to_sample={_NR_ENDPOINTS} was never reached; observed {sorted(observed_counts)}"
    )
    assert min(observed_counts) >= 1
    assert max(observed_counts) <= _NR_ENDPOINTS


def test_forecasting_multiple_samples_per_date_use_different_endpoints():
    """
    Verify that several samples for one split date do not repeat the same endpoints.

    The pool of eligible variables used to be read once, before the sampling loop, so removing the
    already-used variables had no effect and every sample could draw the same endpoints.
    """

    cfg = _base_config()
    _, events, patient_data, dm, all_possible_split_dates = _multi_endpoint_setup(cfg)
    splitter = _multi_endpoint_splitter(cfg, dm, 5, 5)

    np.random.seed(42)
    date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=3,
        override_variables_to_predict=None,
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )

    assert len(date_splits) == 3
    sampled_sets = [set(option.sampled_variables) for option in date_splits]
    for first_idx in range(len(sampled_sets)):
        assert len(sampled_sets[first_idx]) == 5
        for second_idx in range(first_idx + 1, len(sampled_sets)):
            assert sampled_sets[first_idx].isdisjoint(sampled_sets[second_idx]), (
                f"Samples {first_idx} and {second_idx} share endpoints: "
                f"{sampled_sets[first_idx] & sampled_sets[second_idx]}"
            )


def test_forecasting_stops_when_endpoints_are_exhausted():
    """Verify that no empty extra samples are produced once all endpoints have been used."""

    cfg = _base_config()
    _, events, patient_data, dm, all_possible_split_dates = _multi_endpoint_setup(cfg, nr_endpoints=4)
    splitter = _multi_endpoint_splitter(cfg, dm, 2, 2)

    np.random.seed(42)
    date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=10,  # far more than the 2 samples the 4 endpoints allow
        override_variables_to_predict=None,
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=False,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )

    assert len(date_splits) == 2
    assert set().union(*[set(option.sampled_variables) for option in date_splits]) == {
        f"endpoint_{idx:02d}" for idx in range(4)
    }


def test_forecasting_warns_when_fewer_endpoints_than_requested(caplog):
    """Verify that under-sampling is reported instead of silently returning fewer endpoints."""

    cfg = _base_config()
    _, events, patient_data, dm, all_possible_split_dates = _multi_endpoint_setup(cfg, nr_endpoints=3)
    splitter = _multi_endpoint_splitter(cfg, dm, _NR_ENDPOINTS, _NR_ENDPOINTS)

    np.random.seed(42)
    with caplog.at_level("WARNING"):
        date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
            curr_date=_BASE_DATE,
            nr_samples=1,
            override_variables_to_predict=None,
            events=events,
            all_possible_split_dates=all_possible_split_dates,
            apply_filtering=False,
            override_split_dates=None,
            patient_data=patient_data,
            lot_date=_BASE_DATE,
        )

    assert len(date_splits) == 1
    assert len(date_splits[0].sampled_variables) == 3
    assert any("min_nr_variables_to_sample" in record.message for record in caplog.records)


# ────────────────────────────────────────────────────────────────────────────
# Outlier filtering with many endpoints of mixed type
# ────────────────────────────────────────────────────────────────────────────


def _make_variable_stats(cfg, entries):
    """
    Build a minimal variable_stats frame.

    Parameters
    ----------
    entries : list[tuple]
        Each entry is ``(event_name, is_numeric, mean, std)``.
    """

    return pd.DataFrame(
        [
            {
                cfg.event_name_col: name,
                "is_numeric": is_numeric,
                "mean": mean,
                "std": std,
            }
            for name, is_numeric, mean, std in entries
        ]
    )


def test_filter_3_sigma_keeps_variables_without_statistics():
    """
    Verify that outlier filtering does not destroy endpoints which have no usable statistics.

    With many endpoints it is common that some have fewer than min_num_samples_for_statistics
    samples (so no statistics row at all) or are categorical (statistics row with NaN mean/std).
    Previously the former raised an IndexError and the latter had its whole target dropped, because
    clipping against NaN produced NaN which was then dropped.
    """

    cfg = _base_config()
    events = _make_events(
        cfg,
        [
            (0, "lot", "lot_marker", "LoT Start"),
            (0, "lab", "hemoglobin", "13.0"),
            (0, "lab", "rare_lab", "5.0"),
            (0, "lab", "ecog_status", "1"),
            (30, "lab", "hemoglobin", "99.0"),  # extreme outlier -> must be clipped
            (30, "lab", "rare_lab", "5.5"),  # no statistics row -> must be kept as is
            (30, "lab", "ecog_status", "2"),  # categorical -> must be kept as is
            (200, "progression", "progression_marker", "progressed"),
        ],
    )
    patient_data = {"events": events, "constant": _make_constant(cfg)}
    dm = _make_dm_stub(cfg, {"hemoglobin": "numeric", "rare_lab": "numeric", "ecog_status": "categorical"})
    endpoints = ["hemoglobin", "rare_lab", "ecog_status"]
    all_possible_split_dates = pd.DataFrame(
        {
            cfg.date_col: [_BASE_DATE] * 3,
            cfg.event_name_col: endpoints,
            cfg.event_category_col: ["lab"] * 3,
            "lot_date": [_BASE_DATE] * 3,
        }
    )

    splitter = _multi_endpoint_splitter(cfg, dm, 3, 3)
    #: hemoglobin has usable statistics, ecog_status is categorical (NaN stats), rare_lab has none
    splitter.variable_stats = _make_variable_stats(
        cfg,
        [
            ("hemoglobin", True, 13.0, 0.5),
            ("ecog_status", False, np.nan, np.nan),
        ],
    )

    np.random.seed(42)
    date_splits, _, _, _ = splitter._generate_variable_splits_for_date(
        curr_date=_BASE_DATE,
        nr_samples=1,
        override_variables_to_predict=endpoints,
        events=events,
        all_possible_split_dates=all_possible_split_dates,
        apply_filtering=True,
        override_split_dates=None,
        patient_data=patient_data,
        lot_date=_BASE_DATE,
    )

    assert len(date_splits) == 1
    target = date_splits[0].target_events_after_split

    #: all three endpoints survive the filtering
    assert set(target[cfg.event_name_col].unique()) == set(endpoints), (
        f"Endpoints were lost during outlier filtering: {target[cfg.event_name_col].unique().tolist()}"
    )

    #: hemoglobin was clipped to mean + 3 * std
    hemoglobin_value = float(target[target[cfg.event_name_col] == "hemoglobin"][cfg.event_value_col].iloc[0])
    assert hemoglobin_value == pytest.approx(13.0 + 3 * 0.5)

    #: the untouched endpoints keep their original values
    assert str(target[target[cfg.event_name_col] == "rare_lab"][cfg.event_value_col].iloc[0]) == "5.5"
    assert str(target[target[cfg.event_name_col] == "ecog_status"][cfg.event_value_col].iloc[0]) == "2"
