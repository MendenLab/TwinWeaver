from twinweaver import (
    DataSplitterForecasting,
    DataManager,
    DataSplitterEvents,
    DataSplitter,
    ConverterInstruction,
    Config,
)
import pandas as pd


class ConvertToText:
    def __init__(
        self,
    ):
        # Set splitting and predictions
        self.config = Config()
        self.config.split_event_category = "lot"
        self.config.event_category_forecast = ["lab"]
        self.config.data_splitter_events_variables_category_mapping = {
            "death": "death",
            "progression": "next progression",  # Custom name in prompt: "next progression" instead of "progression"
        }

        # Set constant
        self.config.constant_columns_to_use = [
            "birthyear",
            "gender",
            "histology",
            "smoking_history",
        ]  # Manually set from constant
        self.config.constant_birthdate_column = "birthyear"

        # Load data
        df_events = pd.read_csv("./examples/example_data/events.csv")
        df_constant = pd.read_csv("./examples/example_data/constant.csv")
        df_constant_description = pd.read_csv("./examples/example_data/constant_description.csv")

        # Init data managers
        self.dm = DataManager(config=self.config)
        self.dm.load_indication_data(
            df_events=df_events, df_constant=df_constant, df_constant_description=df_constant_description
        )
        self.dm.process_indication_data()
        self.dm.setup_unique_mapping_of_events()
        self.dm.setup_dataset_splits()
        self.dm.infer_var_types()

        data_splitter_events = DataSplitterEvents(self.dm, config=self.config)
        data_splitter_events.setup_variables()
        data_splitter_forecasting = DataSplitterForecasting(data_manager=self.dm, config=self.config)
        data_splitter_forecasting.setup_statistics()

        # Use the unified DataSplitter API that combines both splitters
        self.data_splitter = DataSplitter(
            data_splitter_events=data_splitter_events,
            data_splitter_forecasting=data_splitter_forecasting,
        )

        self.converter = ConverterInstruction(
            nr_tokens_budget_total=8192,
            config=self.config,
            dm=self.dm,
        )

    def convert_full_to_string_for_one_patient(self, patientid, override_events_or_forecasting="forecasting"):
        patient_data = self.dm.get_patient_data(patientid)
        patient_data["events"] = patient_data["events"].sort_values("date")

        # To simulate that we only have input, half the events
        patient_data["events"] = patient_data["events"].iloc[: int(len(patient_data["events"]) / 2)]

        # Use the unified DataSplitter API for inference
        forecast_split, events_split = self.data_splitter.get_splits_from_patient_inference(
            patient_data,
            inference_type=override_events_or_forecasting,
            forecasting_override_variables_to_predict=["Neutrophils"]
            if override_events_or_forecasting != "events"
            else None,
            events_override_category="death" if override_events_or_forecasting != "forecasting" else None,
            events_override_observation_time_delta=pd.Timedelta(weeks=52)
            if override_events_or_forecasting != "forecasting"
            else None,
        )

        # Set which weeks to predict for forecasting (if applicable)
        forecasting_times_to_predict = None
        if forecast_split is not None:
            forecasting_times_to_predict = {
                "Neutrophils": [1, 2, 8, 11],
            }

        # Convert to text
        converted = self.converter.forward_conversion_inference(
            forecasting_split=forecast_split,
            forecasting_future_weeks_per_variable=forecasting_times_to_predict,
            event_split=events_split,
            custom_tasks=None,
        )
        return converted


################################### Running the example #######################################
converter = ConvertToText()

# Example on how to run conversion for inference (i.e. we do not have target)
# Here we predict 52 week survival (as an event), and no forecasting

# NOTE: run this from the root folder of twinweaver

all_patientids = converter.dm.all_patientids.copy()
all_patientids = all_patientids[:10]


for idx, patientid in enumerate(all_patientids):
    print(idx)

    #: go through all patients and convert them
    patient_data = converter.convert_full_to_string_for_one_patient(
        patientid, override_events_or_forecasting="forecasting"
    )
    print(patient_data["instruction"])


print("Finished")
