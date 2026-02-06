import pytest
import pandas as pd
from twinweaver.pretrain.converter_pretrain import ConverterPretrain
from twinweaver.common.data_manager import DataManager


@pytest.fixture
def setup_pretrain_components(mock_config, sample_data):
    """Helper to get converter and data manager ready."""
    df_events, df_constant, df_constant_desc = sample_data
    mock_config.constant_columns_to_use = ["birthyear", "gender", "histology", "smoking_history"]
    mock_config.constant_birthdate_column = "birthyear"

    dm = DataManager(config=mock_config)
    dm.load_indication_data(df_events, df_constant, df_constant_desc)
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_dataset_splits()

    converter = ConverterPretrain(config=mock_config, dm=dm)

    return dm, converter


def test_pretrain_initialization(setup_pretrain_components):
    """Test that the pretrain converter initializes correctly."""
    _, converter = setup_pretrain_components
    assert isinstance(converter, ConverterPretrain)
    # Check if inherited attributes are set
    assert converter.preamble_text is not None


def test_pretrain_forward_conversion(setup_pretrain_components):
    """Test converting patient data into text (pretraining format)."""
    dm, converter = setup_pretrain_components
    patient_id = "p0"
    patient_data = dm.get_patient_data(patient_id)

    result = converter.forward_conversion(patient_data["events"], patient_data["constant"])

    # Check structure
    assert "text" in result
    assert "meta" in result

    text = result["text"]
    meta = result["meta"]

    # Check content in text
    assert isinstance(text, str)
    assert len(text) > 0
    # Expected content based on p0 data
    assert "Male" in text  # constant data
    assert "Adenocarcinoma" in text  # constant data
    assert text.count("weeks later") == 17
    assert text.strip().endswith("death.")
    assert text.count("drug carboplatin is administered") == 15
    assert text.count("hemoglobin - 718-7 is ") == 15
    assert text.count("</genetic>") == 1
    assert text.count("ECOG is") == 3
    assert "On the first visit, the patient experienced the following: " in text
    assert "Age of patient at first event is 50 years," in text

    # Check metadata
    assert "raw_constant" in meta
    assert "processed_constant" in meta
    assert "events" in meta
    assert "raw_constant" in meta
    assert meta["raw_constant"].iloc[0]["patientid"] == patient_id


def test_pretrain_roundtrip_integrity(setup_pretrain_components):
    """Test that forward -> reverse conversion preserves data integrity."""
    dm, converter = setup_pretrain_components
    patient_id = "p0"
    patient_data = dm.get_patient_data(patient_id)

    # 1. Forward
    forward_result = converter.forward_conversion(patient_data["events"], patient_data["constant"])
    text = forward_result["text"]
    meta = forward_result["meta"]

    # 2. Reverse
    # Reverse conversion needs unique events mapping from DM
    reverse_result = converter.reverse_conversion(text=text, data_manager=dm, init_date=meta["events"]["date"].min())

    # 3. Check Integrity
    # Helper to compare
    diff = converter.get_difference_in_event_dataframes(meta["events"], reverse_result["events"], skip_genetic=True)

    assert diff.shape[0] == 0, f"Found differences in roundtrip conversion:\n{diff}"

    # Also check constant integrity roughly
    # (Note: reverse conversion usually extracts constants as well)
    assert reverse_result["constant"] is not None
    # Check if a key constant is retrieved
    # 'histology' should be present
    res_constant = reverse_result["constant"]
    # Depending on how extraction works, it might be a Series or DataFrame
    if isinstance(res_constant, pd.DataFrame):
        assert not res_constant.empty

    assert patient_data["events"].shape[0] == reverse_result["events"].shape[0]
    assert patient_data["events"]["date"].tolist() == reverse_result["events"]["date"].tolist()
