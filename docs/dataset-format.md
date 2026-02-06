# Dataset Format

TwinWeaver expects three primary dataframes (or CSV files) as input. Example files can be found in [`examples/example_data/`](https://github.com/MendenLab/TwinWeaver/tree/main/examples/example_data).

## 1. Longitudinal Events (`events.csv`)

Contains time-varying clinical data where each row represents a single event.

| Column | Description |
|--------|-------------|
| `patientid` | Unique identifier for the patient |
| `date` | Date of the event |
| `event_descriptive_name` | Human-readable name used in the text output |
| `event_category` | *(Optional)* Category (e.g., `lab`, `drug`), used for determining splits & tasks |
| `event_name` | *(Optional)* Specific event identifier |
| `event_value` | Value associated with the event |
| `meta_data` | *(Optional)* Additional metadata |
| `source` | *(Optional)* Modality of data - default to "events", alternatively "genetic" |

**Example:**

```csv
patientid,date,event_descriptive_name,event_category,event_name,event_value,meta_data,source
patient_001,2024-01-15,Hemoglobin,lab,HGB,12.5,,clinical
patient_001,2024-01-15,White Blood Cells,lab,WBC,7.2,,clinical
patient_001,2024-02-01,Chemotherapy Started,treatment,CHEMO,1,,clinical
```

---

## 2. Patient Constants (`constant.csv`)

Contains static patient information (demographics, baseline characteristics). One row per patient.

| Column | Description |
|--------|-------------|
| `patientid` | Unique identifier for the patient |
| `birthyear` | *(example)* Patient's year of birth |
| `gender` | *(example)* Patient's gender |
| `...` | Any other static patient attributes |

**Example:**

```csv
patientid,birthyear,gender,diagnosis_stage
patient_001,1965,Female,Stage II
patient_002,1978,Male,Stage III
```

---

## 3. Constant Descriptions (`constant_description.csv`)

Maps columns in the `constant` table to human-readable descriptions for the text prompt.

| Column | Description |
|--------|-------------|
| `variable` | Name of the column in the constant table |
| `comment` | Description of the variable for the text prompt |

**Example:**

```csv
variable,comment
birthyear,Year of birth
gender,Patient gender
diagnosis_stage,Cancer stage at diagnosis
```

---

## Best Practices for Data Processing

When transforming raw clinical data into TwinWeaver format, following these principles will help you get the most out of your data.

### 1. Prefer Events Over Constants

**Key principle**: Put as much data as possible into the events table. Only truly immutable patient characteristics should go into constants.

Even data that appears "constant" is often better represented as events because:

- **It has a specific date** when it was measured (e.g., biomarker test date)
- **It could change over time** (e.g., acquired resistance mutations, re-staging)
- **Temporal context matters** clinically (when was this information known?)

Examples:
| Data Type | Recommended Table | Rationale |
|-----------|------------------|-----------|
| Birth year, biological sex | `constant` | Truly immutable |
| Biomarker results (EGFR, ALK, PD-L1) | `events` | Has test date, could change |
| Cancer stage | `events` | Stage at diagnosis date, may be re-staged |
| Diagnosis information | `events` | Occurred at a specific date |
| Lab values, vitals | `events` | Longitudinal measurements |
| Treatment administrations | `events` | Time-varying interventions |
| Death, progression | `events` | Time-to-event outcomes |

### 2. Include All Available Data First

Start by including everything, then trim during data generation if needed:

- The `ConverterInstruction` token budget automatically controls output length
- The framework prioritizes recent and relevant events
- You can always exclude data later, but you can't include what wasn't captured

### 3. Use Consistent Event Naming

Standardize your event names and categories:

```python
# Good: Consistent naming convention
event_name = "hemoglobin_-_718-7"  # Includes LOINC code for clarity
event_descriptive_name = "hemoglobin - 718-7"  # Human-readable version

# Avoid: Inconsistent naming
event_name = "Hgb"  # One record
event_name = "hemoglobin"  # Another record
event_name = "HGB"  # Yet another
```

### 4. Structure Event Categories Meaningfully

Choose event categories that align with your modeling objectives:

| Category | Description | Example Events |
|----------|-------------|----------------|
| `lab` | Laboratory test results | hemoglobin, platelets, creatinine |
| `drug` | Drug administrations | pembrolizumab, carboplatin |
| `lot` | Line of therapy markers | treatment start, line number |
| `death` | Mortality events | death |
| `response` | Treatment response | RECIST response, progression |
| `staging` | Cancer staging | stage, TNM classification |
| `basic_biomarker` | Molecular markers | EGFR, ALK, KRAS |

### 5. Use Preprocessing Helper Functions

TwinWeaver provides helper functions to analyze and prepare your data:

```python
from twinweaver import (
    identify_constant_and_changing_columns,
    aggregate_events_to_weeks,
)

# Identify which columns are truly constant vs. changing over time
constant_cols, changing_cols = identify_constant_and_changing_columns(
    df, date_column="visit_date", patientid_column="patient_id"
)

# Aggregate frequent measurements to reduce noise
df_aggregated = aggregate_events_to_weeks(
    df_events,
    patientid_column="patientid",
    date_column="date",
    event_name_column="event_name",
    event_value_column="event_value",
)
```

### 6. Validate Your Data Before Training

Always validate your data format before proceeding:

```python
def validate_twinweaver_format(df_events, df_constant, df_constant_description):
    """Validate that dataframes conform to TwinWeaver requirements."""
    issues = []

    # Check required columns
    events_required = ["patientid", "date", "event_category", "event_name",
                       "event_value", "event_descriptive_name"]
    for col in events_required:
        if col not in df_events.columns:
            issues.append(f"df_events missing column: {col}")

    # Check patient ID consistency
    events_patients = set(df_events["patientid"].unique())
    constant_patients = set(df_constant["patientid"].unique())
    if events_patients != constant_patients:
        issues.append("Patient IDs don't match between events and constants")

    return len(issues) == 0, issues
```

### 7. Handle Time-to-Event Outcomes Properly

Death, progression, and other time-to-event outcomes should be represented as events with a specific date:

```python
# Death event
{
    "patientid": "PT001",
    "date": "2021-02-10",  # Date of death
    "event_category": "death",
    "event_name": "death",
    "event_value": "Yes",
    "event_descriptive_name": "Death",
}
```

!!! note "Censored Patients"
    For patients who are alive (censored), simply don't include a death event. The absence of a death event indicates the patient was alive at last follow-up.

---

## Loading Data

Data can be loaded as pandas DataFrames directly:

```python
import pandas as pd
from twinweaver import DataManager, Config

# Load your data
df_events = pd.read_csv("events.csv")
df_constant = pd.read_csv("constant.csv")
df_constant_description = pd.read_csv("constant_description.csv")

# Initialize configuration
config = Config()

# <---------------------- CRITICAL CONFIGURATION ---------------------->
# 1. Event category used for data splitting (e.g., split data around Lines of Therapy 'lot')
# Has to be set for all instruction tasks
config.split_event_category = "lot"

# 2. List of event categories we want to forecast (e.g., forecasting 'lab' values)
# Only needs to be set if you want to forecast variables
config.event_category_forecast = ["lab"]

# 3. Mapping of specific time to events to predict (e.g., we want to predict 'death' and 'progression')
# Only needs to be set if you want to do time to event prediction
config.data_splitter_events_variables_category_mapping = {
    "death": "death",
    "progression": "next progression",  # Custom name in prompt
}

# Initialize DataManager
dm = DataManager(config=config)
dm.load_indication_data(
    df_events=df_events,
    df_constant=df_constant,
    df_constant_description=df_constant_description
)
```

!!! tip "Configuration Parameters"
    - **`split_event_category`**: The event category used to anchor split points for generating training samples (required for instruction tuning)
    - **`event_category_forecast`**: Which event categories to forecast as time-series values
    - **`data_splitter_events_variables_category_mapping`**: Maps event names to prediction tasks (e.g., survival, progression)

See the [Raw Data Preprocessing Tutorial](examples/data_preprocessing/raw_data_preprocessing.ipynb) for transforming raw clinical data into TwinWeaver format, or the [Data Preparation Tutorial](examples/01_data_preparation_for_training.ipynb) for a complete walkthrough of instruction-tuning data generation.
