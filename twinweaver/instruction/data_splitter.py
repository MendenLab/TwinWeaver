from datetime import datetime
import pandas as pd

from twinweaver.instruction.data_splitter_events import DataSplitterEvents, DataSplitterEventsGroup
from twinweaver.instruction.data_splitter_forecasting import DataSplitterForecasting, DataSplitterForecastingGroup


class DataSplitter:
    """
    Combines both data splitters into one interface for easier usage.
    For more advanced use cases, the individual data splitters can still be used directly.
    """

    def __init__(self, data_splitter_events: DataSplitterEvents, data_splitter_forecasting: DataSplitterForecasting):
        self.data_splitter_events = data_splitter_events
        self.data_splitter_forecasting = data_splitter_forecasting

    def get_splits_from_patient_with_target(
        self,
        patient_data: dict,
        max_num_splits_per_split_event: int = 1,
        forecasting_nr_samples_per_split: int = 1,
        events_max_nr_samples_per_split: int = 1,
        forecasting_filter_outliers: bool = False,
        forecasting_override_categories_to_predict: list[str] = None,
        forecasting_override_variables_to_predict: list[str] = None,
        forecasting_override_split_dates: list[datetime] = None,
        events_override_category: str = None,
        events_override_observation_time_delta: pd.Timedelta = None,
    ) -> tuple[list[DataSplitterForecastingGroup], list[DataSplitterEventsGroup]]:
        """
        Generates both forecasting and event prediction splits for a patient, ensuring proper alignment.

        This value uses the forecasting splitter to determine candidate split dates (based on LoT
        or overrides), which are then passed to the event prediction splitter to ensure both tasks
        use the same anchor points in time. This is critical for multitasking or consistent
        evaluation.

        Parameters
        ----------
        patient_data : dict
            Dictionary containing the patient's data ('events' and 'constant').
        max_num_splits_per_split_event : int
            Maximum number of random split dates to select per Line of Therapy. Defaults to 1.
        forecasting_nr_samples_per_split : int
            Number of forecasting task variants (variable subsets) to generate per split date. Defaults to 1.
        events_max_nr_samples_per_split : int
            Maximum number of event prediction tasks to generate per split date. Defaults to 1.
        forecasting_filter_outliers : bool
            Whether to apply outlier filtering (e.g., 3-sigma) to target values in forecasting tasks.
            Defaults to False.
        forecasting_override_categories_to_predict : list[str], optional
            Force forecasting of all variables in these categories.
        forecasting_override_variables_to_predict : list[str], optional
            Force forecasting of these specific variables.
        forecasting_override_split_dates : list[datetime], optional
            Force usage of these specific split dates.
        events_override_category : str, optional
            Force event prediction for this specific event category.
        events_override_observation_time_delta : pd.Timedelta, optional
            Force a specific prediction window duration for event tasks.

        Returns
        -------
        tuple
            A tuple containing three elements:
            1. forecasting_splits: list[DataSplitterForecastingGroup]
               List of generated forecasting split groups.
            2. events_splits: list[DataSplitterEventsGroup]
               List of generated event prediction split groups, corresponding to the forecasting splits.
            3. reference_dates: pd.DataFrame
               DataFrame containing the split dates and LoT dates used.
        """
        # Process forecasting splits
        forecasting_splits, reference_dates = self.data_splitter_forecasting.get_splits_from_patient(
            patient_data,
            nr_samples_per_split=forecasting_nr_samples_per_split,
            include_metadata=True,
            max_num_splits_per_split_event=max_num_splits_per_split_event,
            filter_outliers=forecasting_filter_outliers,
            override_categories_to_predict=forecasting_override_categories_to_predict,
            override_variables_to_predict=forecasting_override_variables_to_predict,
            override_split_dates=forecasting_override_split_dates,
        )

        # Process event prediction splits
        events_splits = self.data_splitter_events.get_splits_from_patient(
            patient_data,
            reference_split_dates=reference_dates,
            max_nr_samples_per_split=events_max_nr_samples_per_split,
            override_category=events_override_category,
            override_observation_time_delta=events_override_observation_time_delta,
        )

        #: return both, since we want to be able to still have the flexibility to use both splitters directly
        return forecasting_splits, events_splits, reference_dates

    def get_splits_from_patient_inference(
        self,
        patient_data: dict,
        inference_type: str = "both",
        forecasting_override_variables_to_predict: list[str] = None,
        events_override_category: str = None,
        events_override_observation_time_delta: pd.Timedelta = None,
    ):
        """
        Generates a single split for inference based on the latest available data.

        This method assumes the inference should occur at the last recorded date in the
        patient's event history. It generates a single split (forecasting, events, or both)
        anchored at this date. This is typically used for generating predictions on new data.
        Target values will not be available or filtered.

        Parameters
        ----------
        patient_data : dict
            Dictionary containing the patient's data. 'events' dataframe must be present.
        inference_type : str
            The type of inference task to generate: 'forecasting', 'events', or 'both'.
            Defaults to "both".
        forecasting_override_variables_to_predict : list[str], optional
            List of variables to generate forecasts for. If None, variables might be sampled
            (though sampling behavior depends on implementation when no target is present).
        events_override_category : str, optional
            The event category to predict (e.g., 'death', 'progression').
        events_override_observation_time_delta : pd.Timedelta, optional
            The time window for the event prediction.

        Returns
        -------
        tuple
            A tuple containing:
            1. forecast_split: DataSplitterForecastingOption or None
               The generated forecasting option, or None if inference_type is 'events'.
            2. events_split: DataSplitterEventsOption or None
               The generated event prediction option, or None if inference_type is 'forecasting'.
        """
        # assume last date in events is the split date that we're interested in
        patient_data["events"] = patient_data["events"].sort_values(by=self.data_splitter_events.config.date_col)
        split_date = patient_data["events"][self.data_splitter_events.config.date_col].iloc[-1]

        #: generate forecasting split
        if inference_type == "both" or inference_type == "forecasting":
            forecast_splits = self.data_splitter_forecasting.get_splits_from_patient(
                patient_data,
                nr_samples_per_split=1,
                filter_outliers=False,  # Since no filtering needed, since no target exists
                override_split_dates=[split_date],
                override_variables_to_predict=forecasting_override_variables_to_predict,
            )
            # The first one is the only one
            forecast_split = forecast_splits[0][0]
        else:
            forecast_split = None

        #: generate event split
        if inference_type == "both" or inference_type == "events":
            events_splits = self.data_splitter_events.get_splits_from_patient(
                patient_data,
                max_nr_samples_per_split=1,
                override_split_dates=[split_date],
                override_category=events_override_category,
                override_observation_time_delta=events_override_observation_time_delta,
            )
            # The first one is the only one
            events_split = events_splits[0][0]
        else:
            events_split = None

        return forecast_split, events_split
