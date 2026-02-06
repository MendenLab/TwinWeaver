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


Generally, we prefer to keep as much as possible inot the long events table, and only put things into constant that cannot go anywhere else.


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

See the [Data Preparation Tutorial](examples/01_data_preparation_for_training.ipynb) for a complete walkthrough.
