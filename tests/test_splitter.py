import pytest
import pandas as pd
from twinweaver.common.data_manager import DataManager
from twinweaver.instruction.data_splitter_events import DataSplitterEvents
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting
from twinweaver.instruction.data_splitter import DataSplitter


@pytest.fixture
def initialized_dm(mock_config, sample_data):
    """Fixture that returns a fully processed DataManager."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_dataset_splits()
    dm.infer_var_types()
    return dm


def test_splitter_forecasting_statistics(initialized_dm, mock_config):
    """Test that forecasting splitter can calculate statistics."""
    splitter_forecast = DataSplitterForecasting(data_manager=initialized_dm, config=mock_config)

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
    splitter_events = DataSplitterEvents(initialized_dm, config=mock_config)
    splitter_events.setup_variables()

    splitter_forecast = DataSplitterForecasting(data_manager=initialized_dm, config=mock_config)
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
    assert not e_split.event_occurred
    assert e_split.observation_end_date == pd.Timestamp("2016-11-23 00:00:00")

    # Check specifics of f_split - all calculated manually given the random seed and sample data
    assert f_split.sampled_variables.tolist() == ["hemoglobin_-_718-7"]
    assert f_split.target_events_after_split.shape[0] == 4  # 4 hemoglobin measurements after split
    assert f_split.target_events_after_split["date"].min() == pd.Timestamp("2015-07-29 00:00:00")
    assert f_split.target_events_after_split["date"].max() == pd.Timestamp("2015-09-30 00:00:00")


def test_inference_split(initialized_dm, mock_config):
    """Test generating an inference split (last date)."""
    splitter_events = DataSplitterEvents(initialized_dm, config=mock_config)
    splitter_events.setup_variables()
    splitter_forecast = DataSplitterForecasting(data_manager=initialized_dm, config=mock_config)

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
    assert e_split.sampled_category == "death"
    assert e_split.observation_end_date == pd.Timestamp("2016-08-26 00:00:00")
    assert e_split.event_censored == "end_of_data"
    assert not e_split.event_occurred
