from twinweaver import (
    DataSplitterForecasting,
    DataManager,
    DataSplitterEvents,
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

        self.data_splitter_events = DataSplitterEvents(self.dm, config=self.config)
        self.data_splitter_events.setup_variables()
        self.data_splitter_forecasting = DataSplitterForecasting(data_manager=self.dm, config=self.config)
        self.data_splitter_forecasting.setup_statistics()
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

        # Here then split date
        split_date = patient_data["events"]["date"].iloc[-1]

        #: generate event split - NOTE: this if statement is only to exemplify both cases!
        if override_events_or_forecasting == "events":
            ####### Example if we want to override for events

            events_splits = self.data_splitter_events.get_splits_from_patient(
                patient_data,
                max_nr_samples=1,
                override_split_dates=[split_date],
                override_category="death",
                override_end_week_delta=52,
            )
            # We just pick the first one
            events_split = events_splits[0][0]

            #: no forecasting split
            forecast_split = None
            forecasting_times_to_predict = None
        else:
            ####### Example if we want to override for forecasting

            #: generate forecasting split
            forecast_splits = self.data_splitter_forecasting.get_splits_from_patient(
                patient_data,
                nr_samples_per_split=1,
                filter_outliers=False,
                override_split_dates=[split_date],
                override_variables_to_predict=["Neutrophils"],
            )
            # We just pick the first one
            forecast_split = forecast_splits[0][0]

            # We set which weeks to predict
            forecasting_times_to_predict = {
                "Neutrophils": [1, 2, 8, 11],
            }

            #: no events split
            events_split = None

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
