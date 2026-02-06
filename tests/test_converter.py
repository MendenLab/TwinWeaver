import pytest
import pandas as pd
from twinweaver.instruction.converter_instruction import ConverterInstruction
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting
from twinweaver.instruction.data_splitter_events import DataSplitterEvents
from twinweaver.instruction.data_splitter import DataSplitter
from twinweaver.common.data_manager import DataManager


@pytest.fixture
def setup_components(mock_config, sample_data):
    """Helper to get converter and splitters ready."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.data_splitter_events_variables_category_mapping = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_dataset_splits()
    dm.infer_var_types()

    splitter_events = DataSplitterEvents(dm, config=mock_config)
    splitter_events.setup_variables()

    splitter_forecast = DataSplitterForecasting(data_manager=dm, config=mock_config)
    splitter_forecast.setup_statistics()

    data_splitter = DataSplitter(splitter_events, splitter_forecast)

    converter = ConverterInstruction(
        nr_tokens_budget_total=4096, config=mock_config, dm=dm, variable_stats=splitter_forecast.variable_stats
    )

    return dm, data_splitter, converter


def test_forward_conversion_training(setup_components):
    """Test converting a split into a training prompt/completion pair."""
    dm, data_splitter, converter = setup_components
    patient_data = dm.get_patient_data("p0")

    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    # Perform conversion
    result = converter.forward_conversion(
        forecasting_splits=f_splits[0],
        event_splits=e_splits[0],
        override_mode_to_select_forecasting="both",  # Test both QA and regular forecasting
    )

    instruction = result["instruction"]
    answer = result["answer"]
    meta = result["meta"]

    # 1. Check Instruction Content
    assert "Starting with demographic data:" in instruction
    assert "p0" in str(meta["patientid"])  # Should match patient id
    assert "Task" in instruction  # Should be multi-task

    # Specific content checks
    assert "The most recent line of therapy:" in instruction
    assert "LoT - pemetrexed" in instruction  # Known LoT for p0
    assert "The last values of the variables in the input data are:" in instruction
    assert "hemoglobin - 718-7 was 14.13" in instruction  # Known last value for p0
    assert instruction.count("</genetic>") == 2  # Two genetic markers for p0 (main + recap)
    assert instruction.count("ALK is Wild Type,") == 2
    assert instruction.count("PD-L1 Expression (TPS) is 1-49%,") == 2
    assert instruction.count("drug pemetrexed is administered") == 4
    assert "Age of patient at first event is 50 years," in instruction
    assert "Adenocarcinoma" in instruction

    # Specific task checks for this patient (randomly selects forecasting QA, forecasting, no events here)
    assert "Task 1 is forecasting QA:" in instruction
    assert "Task 2 is forecasting:" in instruction
    assert "Task 3" not in instruction  # No events task here
    assert "hemoglobin - 718-7 the future weeks 3, 6, 9, 12" in instruction  # Forecasting QA content
    assert "The possible bins are: a: bin (-inf, 12.56], b: bin (12.56, 12.85],"

    # 2. Check Answer Content
    assert "Task" in answer
    assert len(answer) > 0
    assert answer.count("Task") == 2  # Two tasks here
    assert "Task 1 is forecasting QA:" in answer
    assert "Task 2 is forecasting:" in answer
    assert "hemoglobin - 718-7 is e." in answer  # Known bin for p0
    assert "hemoglobin - 718-7 is 14.01." in answer  # Known value for p0


def test_forward_conversion_inference(setup_components):
    """Test conversion for inference (no target string)."""
    dm, data_splitter, converter = setup_components
    patient_data = dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data, forecasting_override_variables_to_predict=["hemoglobin_-_718-7"]
    )

    result = converter.forward_conversion_inference(
        forecasting_split=f_split,
        forecasting_future_weeks_per_variable={"hemoglobin_-_718-7": [4, 8, 12]},
        event_split=e_split,
    )

    assert result["answer"] is None
    assert "hemoglobin" in result["instruction"]
    assert "future weeks 4, 8, 12" in result["instruction"]
    assert "Task 1 is time to event prediction:" in result["instruction"]  # Events task present
    assert "Task 2 is forecasting:" in result["instruction"]  # Forecasting task present
    # Events are randomly selected since we don't provide override values in splitter above
    assert "whether the following event was censored 93 weeks from the last clinical" in result["instruction"]


def test_reverse_conversion(setup_components):
    """Test that the converter can parse its own output (or similar valid output)."""
    dm, data_splitter, converter = setup_components

    # Simulate a model output string
    # Note: Using exact strings from Config defaults
    task_forecasting = "forecasting:"
    task_events = "time to event prediction:"

    fake_model_output = (
        f"Task 1 is {task_forecasting}\n3 weeks later, the patient visited and experienced the following: "
        "\n\themoglobin - 718-7 is 13.18,\n\tpotassium - 2823-3 is 3.98.\n\n3 weeks later, the patient visited and "
        "experienced the following: \n\themoglobin - 718-7 is 13.3,\n\tpotassium - 2823-3 is 3.91.\n\n"
        f"Task 2 is {task_events}\nHere is the prediction: the event (death) was not censored and did not occur.\n\n"
    )

    # We need a reference split date for forecasting reverse conversion
    ref_date = pd.Timestamp("2016-01-01")

    parsed = converter.reverse_conversion(target_string=fake_model_output, data_manager=dm, split_date=ref_date)

    assert len(parsed) == 2

    # Check Forecasting Parse
    f_res = parsed[0]
    assert f_res["task_type"] == task_forecasting
    assert (
        "hemoglobin" in f_res["result"]["event_name"].iloc[0].lower()
        or "hemoglobin" in f_res["result"]["event_descriptive_name"].iloc[0].lower()
    )
    assert float(f_res["result"]["event_value"].iloc[0]) == 13.18
    assert (
        "potassium" in f_res["result"]["event_name"].iloc[1].lower()
        or "potassium" in f_res["result"]["event_descriptive_name"].iloc[1].lower()
    )
    assert float(f_res["result"]["event_value"].iloc[1]) == 3.98
    assert (f_res["result"]["date"].iloc[0] - ref_date).days == 21  # 3 weeks later
    assert (f_res["result"]["date"].iloc[1] - ref_date).days == 21  # 3 weeks later
    assert (f_res["result"]["date"].iloc[2] - ref_date).days == 42  # 6 weeks later
    assert (f_res["result"]["date"].iloc[3] - ref_date).days == 42  # 6 weeks later
    assert len(f_res["result"]) == 4

    # Check Events Parse
    e_res = parsed[1]
    assert e_res["task_type"] == task_events
    assert e_res["result"]["censoring"].iloc[0].item() is False
    assert e_res["result"]["occurred"].iloc[0].item() is False
