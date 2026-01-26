import pytest
import pandas as pd
from twinweaver.common.config import Config


@pytest.fixture
def mock_config():
    """Returns a Config object with default settings."""
    cfg = Config()
    # Ensure the random seed is fixed for reproducible tests
    cfg.seed = 42
    return cfg


@pytest.fixture
def sample_data():
    # Load the test data
    df_events = pd.read_csv("tests/test_data/test_events.csv")
    df_constant = pd.read_csv("tests/test_data/test_constant.csv")
    df_constant_desc = pd.read_csv("tests/test_data/test_constant_description.csv")

    return df_events, df_constant, df_constant_desc
