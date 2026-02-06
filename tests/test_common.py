import pandas as pd
from twinweaver.common.data_manager import DataManager


def test_config_initialization(mock_config):
    """Test that config initializes with default values."""
    assert mock_config.seed == 42
    assert mock_config.patient_id_col == "patientid"
    # Verify defaults used in the library
    assert mock_config.event_category_lot == "lot"


def test_data_manager_loading(mock_config, sample_data):
    """Test that DataManager loads data correctly."""
    df_events, df_constant, df_constant_desc = sample_data

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)

    assert dm.data_frames["events"].shape == df_events.shape
    assert dm.data_frames["constant"].shape == df_constant.shape

    events_cols = [
        "event_category",
        "event_name",
        "date",
        "event_value",
        "patientid",
        "source",
        "meta_data",
        "event_descriptive_name",
    ]
    for col in events_cols:
        assert col in dm.data_frames["events"].columns

    # Check that 'Unnamed' columns were removed if they existed (implied by logic)
    for col in dm.data_frames["events"].columns:
        assert "Unnamed" not in col


def test_data_manager_processing(mock_config, sample_data):
    """Test the full processing pipeline of DataManager."""
    df_events, df_constant, df_constant_desc = sample_data

    # Override config
    mock_config.split_event_category = "lot"
    mock_config.event_category_forecast = ["lab"]
    mock_config.data_splitter_events_variables_category_mapping = None
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)

    # Run pipeline
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_dataset_splits()
    dm.infer_var_types()

    # 1. Check Date Processing
    assert pd.api.types.is_datetime64_any_dtype(dm.data_frames["events"]["date"])

    # 2. Check Patient Splitting
    assert "data_split" in dm.data_frames["constant"].columns
    splits = dm.data_frames["constant"]["data_split"].unique()
    assert len(splits) > 0

    # 3. Check Variable Inference
    # hemoglobin is numeric in the csv
    hemoglobin_var = "hemoglobin_-_718-7"
    assert hemoglobin_var in dm.variable_types
    assert dm.variable_types[hemoglobin_var] == "numeric"


def test_patient_data_retrieval(mock_config, sample_data):
    """Test retrieving data for a specific patient."""
    df_events, df_constant, df_constant_desc = sample_data
    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()

    patient_id = "p0"
    patient_data = dm.get_patient_data(patient_id)

    assert "events" in patient_data
    assert "constant" in patient_data
    assert len(patient_data["constant"]) == 1
    assert patient_data["constant"]["patientid"].iloc[0] == patient_id
    assert len(patient_data["events"]) == 78  # Manually retrieved from data
    assert patient_data["events"].iloc[0]["event_category"] == "main_diagnosis"  # Manually checked from raw data
    assert patient_data["events"].iloc[-1]["event_category"] == "death"  # Manually checked from raw data
    # Ensure events are sorted
    assert patient_data["events"]["date"].is_monotonic_increasing
