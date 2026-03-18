import pytest
import numpy as np
import pandas as pd
from twinweaver.common.data_manager import DataManager
from twinweaver.instruction.data_splitter_events import DataSplitterEvents
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting
from twinweaver.instruction.data_splitter import DataSplitter


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
    assert e_split.observation_end_date == pd.Timestamp("2016-11-23 00:00:00")
    assert e_split.sampled_category == "death"

    # Check specifics of f_split - all calculated manually given the random seed and sample data
    assert f_split.sampled_variables.tolist() == ["hemoglobin_-_718-7"]
    assert f_split.target_events_after_split.shape[0] == 4  # 4 hemoglobin measurements after split
    assert f_split.target_events_after_split["date"].min() == pd.Timestamp("2015-07-29 00:00:00")
    assert f_split.target_events_after_split["date"].max() == pd.Timestamp("2015-09-30 00:00:00")


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
    assert ref_dates["date"].iloc[0] == pd.Timestamp("2015-07-08")
    assert ref_dates["split_date"].iloc[0] == pd.Timestamp("2015-05-06")

    # Validate forecasting split structure and content
    f_split = forecasting_splits[0][0]
    assert f_split.events_until_split is not None
    assert f_split.constant_data["patientid"].iloc[0] == "p0"
    assert f_split.events_until_split.shape == (32, 8)
    assert f_split.split_date_included_in_input == pd.Timestamp("2015-07-08")
    assert f_split.lot_date == pd.Timestamp("2015-05-06")
    assert f_split.sampled_variables == ["hemoglobin_-_718-7"]

    # Target events should exist and be after the split date
    assert not f_split.target_events_after_split.empty
    assert f_split.target_events_after_split.shape[0] == 4
    assert f_split.target_events_after_split["date"].min() == pd.Timestamp("2015-07-29")
    assert f_split.target_events_after_split["date"].max() == pd.Timestamp("2015-09-30")

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
    assert ref_dates["date"].iloc[0] == pd.Timestamp("2015-07-08")
    assert ref_dates["split_date"].iloc[0] == pd.Timestamp("2015-05-06")

    # Validate events split structure and content
    e_split = events_splits[0][0]
    assert e_split.events_until_split is not None
    assert e_split.constant_data["patientid"].iloc[0] == "p0"
    assert e_split.events_until_split.shape == (32, 8)
    assert e_split.split_date_included_in_input == pd.Timestamp("2015-07-08")
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
    assert e_split.events_until_split["date"].max() == pd.Timestamp("2015-07-08")

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
