import pytest
import pandas as pd
from twinweaver.instruction.converter_instruction import ConverterInstruction
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting
from twinweaver.instruction.data_splitter_events import DataSplitterEvents
from twinweaver.instruction.data_splitter import DataSplitter
from twinweaver.common.data_manager import DataManager
from twinweaver.utils.forecasting_inference import parse_forecasting_results


@pytest.fixture
def setup_components(mock_config, sample_data):
    """Helper to get converter and splitters ready."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"

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
        min_length_to_sample=pd.Timedelta(weeks=1),
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


def test_event_categories_to_exclude_from_input(mock_config, sample_data):
    """Test that event_categories_to_exclude_from_input removes specified categories from generated text."""
    df_events, df_constant, df_constant_desc = sample_data

    # Configure with drug events excluded
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"
    mock_config.event_categories_to_exclude_from_input = ["drug"]

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
        min_length_to_sample=pd.Timedelta(weeks=1),
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
        nr_tokens_budget_total=4096, config=mock_config, dm=dm, variable_stats=splitter_forecast.variable_stats
    )

    patient_data = dm.get_patient_data("p0")
    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    result = converter.forward_conversion(
        forecasting_splits=f_splits[0],
        event_splits=e_splits[0],
        override_mode_to_select_forecasting="both",
    )

    instruction = result["instruction"]

    # Drug events (e.g. "drug pemetrexed is administered") should be excluded
    assert "drug pemetrexed is administered" not in instruction
    assert (
        "drug" not in instruction.lower().split("demographic")[0].split("\n")[-1]
        if "demographic" in instruction
        else True
    )

    # Other event categories should still be present
    assert "Starting with demographic data:" in instruction
    assert "hemoglobin" in instruction  # lab events still present


def test_event_categories_to_exclude_multiple(mock_config, sample_data):
    """Test excluding multiple event categories from input."""
    df_events, df_constant, df_constant_desc = sample_data

    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"
    mock_config.event_categories_to_exclude_from_input = ["drug", "ecog"]

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
        min_length_to_sample=pd.Timedelta(weeks=1),
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
        nr_tokens_budget_total=4096, config=mock_config, dm=dm, variable_stats=splitter_forecast.variable_stats
    )

    patient_data = dm.get_patient_data("p0")
    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    result = converter.forward_conversion(
        forecasting_splits=f_splits[0],
        event_splits=e_splits[0],
        override_mode_to_select_forecasting="both",
    )

    instruction = result["instruction"]

    # Both drug and ecog events should be absent
    assert "drug pemetrexed is administered" not in instruction
    assert "ECOG" not in instruction

    # Lab events should still be present
    assert "hemoglobin" in instruction


def test_event_categories_to_exclude_empty_list(setup_components):
    """Test that an empty exclude list leaves all events intact (default behavior)."""
    dm, data_splitter, converter = setup_components
    assert dm.config.event_categories_to_exclude_from_input == []

    patient_data = dm.get_patient_data("p0")
    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    result = converter.forward_conversion(
        forecasting_splits=f_splits[0],
        event_splits=e_splits[0],
        override_mode_to_select_forecasting="both",
    )

    instruction = result["instruction"]

    # Drug events should be present when nothing is excluded
    assert "drug pemetrexed is administered" in instruction


def test_event_categories_excluded_from_input_but_kept_in_targets(mock_config, sample_data):
    """Exclusion config must affect only input history, not forecasting targets."""
    df_events, df_constant, df_constant_desc = sample_data

    mock_config.split_event_category = "lot"
    # Forecast drug events so target/answer is expected to include drug content
    mock_config.event_category_forecast = ["drug"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"
    mock_config.event_categories_to_exclude_from_input = ["drug"]

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
    dm.infer_var_types()

    splitter_forecast = DataSplitterForecasting(
        data_manager=dm,
        config=mock_config,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_forecast.setup_statistics()

    data_splitter = DataSplitter(data_splitter_forecasting=splitter_forecast)

    converter = ConverterInstruction(
        nr_tokens_budget_total=4096, config=mock_config, dm=dm, variable_stats=splitter_forecast.variable_stats
    )

    patient_data = dm.get_patient_data("p0")
    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    assert e_splits is None
    assert f_splits is not None and len(f_splits) > 0

    result = converter.forward_conversion(
        forecasting_splits=f_splits[0],
        event_splits=None,
        override_mode_to_select_forecasting="forecasting",
    )

    instruction = result["instruction"]
    answer = result["answer"]

    # Input context should not include excluded drug history lines
    input_context = instruction.split("Task 1 is", 1)[0]
    assert "drug pemetrexed is administered" not in input_context

    # Target/answer must still contain the forecasted drug values
    assert "Task 1 is forecasting:" in answer
    assert "drug " in answer.lower()
    assert "is administered" in answer.lower()


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


# ── Tests for forecasting-only and events-only conversion ──────────────────


@pytest.fixture
def setup_forecasting_only(mock_config, sample_data):
    """Helper to set up components with only a forecasting splitter (no events splitter)."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_hold_out_sets(validation_split=0.1, test_split=0.1)
    dm.infer_var_types()

    splitter_forecast = DataSplitterForecasting(
        data_manager=dm,
        config=mock_config,
        max_forecasted_trajectory_length=pd.Timedelta(days=90),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_forecast.setup_statistics()

    data_splitter = DataSplitter(data_splitter_forecasting=splitter_forecast)

    converter = ConverterInstruction(
        nr_tokens_budget_total=4096, config=mock_config, dm=dm, variable_stats=splitter_forecast.variable_stats
    )

    return dm, data_splitter, converter, splitter_forecast


@pytest.fixture
def setup_events_only(mock_config, sample_data):
    """Helper to set up components with only an events splitter (no forecasting splitter)."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.event_category_events_prediction_with_naming = {"death": "death", "progression": "next progression"}
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"

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
        min_length_to_sample=pd.Timedelta(weeks=1),
        max_split_length_after_split_event=pd.Timedelta(days=90),
    )
    splitter_events.setup_variables()

    data_splitter = DataSplitter(data_splitter_events=splitter_events)

    converter = ConverterInstruction(nr_tokens_budget_total=4096, config=mock_config, dm=dm, variable_stats=None)

    return dm, data_splitter, converter


def test_forward_conversion_forecasting_only_training(setup_forecasting_only):
    """Test training conversion with only forecasting splits (no events)."""
    dm, data_splitter, converter, _ = setup_forecasting_only
    patient_data = dm.get_patient_data("p0")

    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    assert e_splits is None  # No events splitter configured
    assert f_splits is not None

    result = converter.forward_conversion(
        forecasting_splits=f_splits[0],
        event_splits=None,
        override_mode_to_select_forecasting="forecasting",
    )

    instruction = result["instruction"]
    answer = result["answer"]

    # Should contain forecasting task but no events task
    assert "Task 1 is forecasting:" in instruction
    assert "time to event prediction:" not in instruction
    assert "Starting with demographic data:" in instruction

    # Answer should contain exactly one task
    assert len(answer) > 0
    assert "Task 1 is forecasting:" in answer
    assert "time to event prediction:" not in answer


def test_forward_conversion_events_only_training(setup_events_only):
    """Test training conversion with only event splits (no forecasting)."""
    dm, data_splitter, converter = setup_events_only
    patient_data = dm.get_patient_data("p0")

    f_splits, e_splits, _ = data_splitter.get_splits_from_patient_with_target(patient_data)

    assert f_splits is None  # No forecasting splitter configured
    assert e_splits is not None

    result = converter.forward_conversion(
        forecasting_splits=None,
        event_splits=e_splits[0],
    )

    instruction = result["instruction"]
    answer = result["answer"]

    # Should contain events task but no forecasting task
    assert "time to event prediction:" in instruction
    assert "forecasting:" not in instruction
    assert "Starting with demographic data:" in instruction

    # Answer should contain events task only
    assert len(answer) > 0
    assert "time to event prediction:" in answer
    assert "forecasting:" not in answer


def test_forward_conversion_inference_forecasting_only(setup_forecasting_only):
    """Test inference conversion with only a forecasting split (no events)."""
    dm, data_splitter, converter, _ = setup_forecasting_only
    patient_data = dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="forecasting",
        forecasting_override_variables_to_predict=["hemoglobin_-_718-7"],
    )

    assert e_split is None
    assert f_split is not None

    result = converter.forward_conversion_inference(
        forecasting_split=f_split,
        forecasting_future_weeks_per_variable={"hemoglobin_-_718-7": [4, 8, 12]},
        event_split=None,
    )

    instruction = result["instruction"]
    assert result["answer"] is None
    assert "hemoglobin" in instruction
    assert "future weeks 4, 8, 12" in instruction
    assert "Task 1 is forecasting:" in instruction
    assert "time to event prediction:" not in instruction


def test_forward_conversion_inference_events_only(setup_events_only):
    """Test inference conversion with only an event split (no forecasting)."""
    dm, data_splitter, converter = setup_events_only
    patient_data = dm.get_patient_data("p0")

    f_split, e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="events",
    )

    assert f_split is None
    assert e_split is not None

    result = converter.forward_conversion_inference(
        forecasting_split=None,
        event_split=e_split,
    )

    instruction = result["instruction"]
    assert result["answer"] is None
    assert "Task 1 is time to event prediction:" in instruction
    assert "forecasting:" not in instruction
    assert "Starting with demographic data:" in instruction


# ── Regression tests for forecasting inference bugs ─────────────────────────


def test_parse_forecasting_results_aggregates_samples(setup_components, capsys):
    """Bug A: sample aggregation must route to the forecasting sub-converter.

    Previously ``parse_forecasting_results`` called
    ``ConverterInstruction.aggregate_multiple_responses`` (which expects
    ``list[list[dict]]``) with a ``list[pd.DataFrame]``, so every aggregation
    threw, was swallowed, and silently fell back to "keep first sample". This
    test feeds two forecasting samples with the same (date, event) but different
    numeric values and asserts the result is their mean (not the first sample).
    """
    dm, _data_splitter, converter = setup_components

    split_date = pd.Timestamp("2016-01-01")

    def _forecast_text(value):
        return (
            "Task 1 is forecasting:\n"
            "3 weeks later, the patient visited and experienced the following: "
            f"\n\themoglobin - 718-7 is {value}.\n\n"
        )

    # Two samples: hemoglobin 13.0 and 15.0 -> mean 14.0
    payload = {
        "patientid": "p0",
        "split_date": split_date,
        "generated_texts": [_forecast_text("13.0"), _forecast_text("15.0")],
    }

    df = parse_forecasting_results([payload], converter, dm, aggregate_samples=True)

    captured = capsys.readouterr()
    assert "aggregation failed" not in captured.out, captured.out

    assert not df.empty
    # Single (date, event) group across the two samples -> one aggregated row.
    hemo = df[df["event_name"].str.contains("hemoglobin", case=False, na=False)]
    assert len(hemo) == 1
    assert float(hemo["event_value"].iloc[0]) == pytest.approx(14.0)
    # No per-sample fan-out should survive aggregation.
    assert "sample_idx" not in df.columns


def test_inference_history_uses_preprocessed_events(setup_forecasting_only):
    """Bug B: inference must render the *preprocessed* history, like training.

    ``forward_conversion_inference`` computed ``patient_history_processed`` and
    then rendered the raw ``patient_history`` instead. ``_preprocess_events``
    drops ``event_categories_to_exclude_from_input`` (and round/strips + sorts),
    so the rendered history differed from the training path. Here we exclude a
    category present in the patient's history and assert it does not leak into
    the inference prompt, and that the inference history block exactly equals the
    rendering of the preprocessed frame.
    """
    dm, data_splitter, converter, _ = setup_forecasting_only

    # Exclude a category that is present in p0's input history.
    converter.config.event_categories_to_exclude_from_input = ["biomarker"]

    patient_data = dm.get_patient_data("p0")
    f_split, _e_split = data_splitter.get_splits_from_patient_inference(
        patient_data,
        inference_type="forecasting",
        forecasting_override_variables_to_predict=["hemoglobin_-_718-7"],
    )

    result = converter.forward_conversion_inference(
        forecasting_split=f_split,
        forecasting_future_weeks_per_variable={"hemoglobin_-_718-7": [4, 8, 12]},
        event_split=None,
    )

    instruction = result["instruction"]
    raw_history = result["meta"]["history_data"]

    # Render both the processed (correct) and raw (buggy) history frames.
    processed = converter._preprocess_events(raw_history.copy())
    expected_history = converter._get_event_string(processed, use_accumulative_dates=False)
    raw_history_str = converter._get_event_string(raw_history.copy(), use_accumulative_dates=False)

    # The exclusion must actually change the rendering (otherwise the test is vacuous).
    assert "Epidermal Growth Factor Receptor" in raw_history_str
    assert "Epidermal Growth Factor Receptor" not in expected_history
    assert expected_history != raw_history_str

    # Post-fix: the prompt contains the preprocessed rendering and the excluded
    # category does not leak in. Pre-fix this fails (raw frame was rendered).
    assert expected_history in instruction
    assert "Epidermal Growth Factor Receptor" not in instruction
