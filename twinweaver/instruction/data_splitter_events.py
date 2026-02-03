import logging
import numpy as np
import pandas as pd

from twinweaver.instruction.data_splitter_base import BaseDataSplitter
from twinweaver.common.data_manager import DataManager
from twinweaver.common.config import Config


class DataSplitterEventsOption:
    """
    A class to hold the options for a single event prediction split.

    Attributes
    ----------
    events_until_split : pd.DataFrame
        DataFrame of events up to the split date.
    constant_data : pd.DataFrame
        DataFrame of constant patient data.
    event_occurred : bool
        Whether the event occurred within the prediction window.
    event_censored : bool
        Whether the event was censored.
    observation_end_date : pd.Timestamp
        The date the event occurred or the end date of the prediction window.
    split_date_included_in_input : pd.Timestamp
        The split date used for input.
    sampled_category : str
        The event category being predicted.
    sampled_category_name : str
        Descriptive name for the sampled category.
    end_date : pd.Timestamp
        The end of the prediction window.
    lot_date : pd.Timestamp
        The Line of Therapy (LoT) start date associated with this split point.
    """

    def __init__(
        self,
        events_until_split: pd.DataFrame,
        constant_data: pd.DataFrame,
        event_occurred: bool,
        event_censored: bool,
        observation_end_date: pd.Timestamp,
        split_date_included_in_input: pd.Timestamp,
        sampled_category: str,
        sampled_category_name: str,
        lot_date: pd.Timestamp,
    ):
        self.events_until_split = events_until_split
        self.constant_data = constant_data
        self.event_occurred = event_occurred
        self.event_censored = event_censored
        self.observation_end_date = observation_end_date
        self.split_date_included_in_input = split_date_included_in_input
        self.sampled_category = sampled_category
        self.sampled_category_name = sampled_category_name
        self.lot_date = lot_date


class DataSplitterEventsGroup:
    """
    A class to hold a group of event prediction options for a single split date.
    Usually one of the elements in this list is then used, e.g. by random
    selection in converter_manual_instruction.
    """

    def __init__(
        self,
        events_options: list[DataSplitterEventsOption] = None,
    ):
        if events_options is None:
            events_options = []
        self.events_options = events_options

    def append(self, option: DataSplitterEventsOption):
        self.events_options.append(option)

    def __len__(self):
        return len(self.events_options)

    def __getitem__(self, index):
        return self.events_options[index]


class DataSplitterEvents(BaseDataSplitter):
    def __init__(
        self,
        data_manager: DataManager,
        config: Config,
        max_length_to_sample: pd.Timedelta = pd.Timedelta(weeks=104),
        min_length_to_sample: pd.Timedelta = pd.Timedelta(weeks=1),
        unit_length_to_sample: str = "weeks",
        max_split_length_after_split_event: pd.Timedelta = pd.Timedelta(days=90),
        max_lookback_time_for_value: pd.Timedelta = pd.Timedelta(days=90),
        max_forecast_time_for_value: pd.Timedelta = pd.Timedelta(days=90),
    ):
        """
        Initialize the DataSplitterEvents class.

        Parameters
        ----------
        data_manager : DataManager
            The data manager to handle data operations.
        config : Config
            Configuration object holding constants.
        max_length_to_sample : pd.Timedelta
            The maximum number of weeks into the future to sample for event prediction.
        min_length_to_sample : pd.Timedelta
            The minimum number of weeks into the future to sample for event prediction.
        unit_length_to_sample : str
            The unit of time for the length to sample (e.g. "weeks").
        max_split_length_after_split_event : pd.Timedelta, optional
            The maximum number of days after the split event (e.g. line of therapy) to consider for split points.
        max_lookback_time_for_value : pd.Timedelta, optional
            The maximum number of days to look back for a value (inherited but not directly used here).
        max_forecast_time_for_value : pd.Timedelta, optional
            The maximum number of days to forecast a value (inherited but not directly used here).
        """
        super().__init__(
            data_manager,
            config,
            max_split_length_after_split_event,
            max_lookback_time_for_value,
            max_forecast_time_for_value,
        )
        self.max_length_to_sample = max_length_to_sample
        self.min_length_to_sample = min_length_to_sample
        self.unit_length_to_sample = unit_length_to_sample

        self.manual_variables_category_mapping = self.config.data_splitter_events_variables_category_mapping

    def setup_variables(self):
        """
        Setup the variables to be used for sampling.
        """

        #: get all categories available
        all_categories = self.dm.data_frames["events"][self.config.event_category_col].unique().tolist()

        #: first look at the manual variables
        self.manual_variables_category_mapping = {
            x: self.manual_variables_category_mapping[x]
            for x in self.manual_variables_category_mapping.keys()
            if x in all_categories
        }

        # Sanity check to ensure we have at least one variable to sample
        if len(self.manual_variables_category_mapping) == 0:
            raise ValueError(
                "No valid event categories found in the data for event prediction splitting. "
                "Check the data or adjust data_splitter_events_variables_category_mapping in Config."
            )

    def _sample_manual_variables(self, events_after_split: pd.DataFrame, override_category: str) -> tuple:
        """
        Sample manual variables from the events occurring after the split date.

        Parameters
        ----------
        events_after_split : pd.DataFrame
            The dataframe containing events that occur after the split date.
        override_category : str
            If provided, the sampling is done for this specific category.

        Returns
        -------
        tuple
            A tuple containing the category of the sampled variable,
            and the descriptive name of the sampled variable.
        """

        if override_category is None:
            #: we need to uniformly sample the exact variable based on category
            category = np.random.choice(list(self.manual_variables_category_mapping.keys()))
        else:
            category = override_category

        # Also return the descriptive name based on category
        next_var_descriptive = self.manual_variables_category_mapping[category]

        # We allow backup categories in case the exact category is not present
        # E.g. in case of progression, try alternatively death, since it is also a progression event
        if category not in events_after_split[self.config.event_category_col].unique():
            if category in self.config.data_splitter_events_backup_category_mapping.keys():
                backup_category = self.config.data_splitter_events_backup_category_mapping[category]
                if backup_category in events_after_split[self.config.event_category_col].unique():
                    category = backup_category

        #: return exact variable
        return category, next_var_descriptive

    def get_splits_from_patient(
        self,
        patient_data: dict,
        max_nr_samples_per_split: int,
        max_num_splits_per_split_event: int = 1,
        reference_split_dates: pd.DataFrame = None,
        override_split_dates: list = None,
        override_category: str = None,
        override_observation_time_delta: pd.Timedelta = None,
    ) -> list[DataSplitterEventsGroup]:
        """
        Generates event prediction tasks (splits) for a given patient.

        For each unique split event (e.g. Line of Therapy, LoT) start date in the patient's history,
        this function potentially selects one or more random split points within a defined
        window after the split event (e.g. LoT start - `max_split_length_after_split_event`). The number of
        split points selected per LoT is controlled by `max_num_splits_per_split_event`.

        If `reference_split_dates` (typically generated by a parallel forecasting
        splitter for consistency) is provided, those exact split dates are used instead
        of random sampling based on the split event. If `override_split_dates` is provided
        (e.g., for inference), those specific dates are used. Only one of
        `reference_split_dates` or `override_split_dates` can be used.

        For each chosen split date (`curr_date`), this method generates multiple event
        prediction tasks (up to `max_nr_samples_per_split`). Each task involves predicting
        a specific event category (e.g., 'death', 'next line of therapy') within a
        randomly determined future time window (`end_week_delta`, up to
        `max_length_to_sample`). The function handles censoring based on
        subsequent events (like next LoT start or death) or end of available data.

        Parameters
        ----------
        patient_data : dict
            A dictionary containing the patient's data. Expected keys:
            'events': pd.DataFrame with patient event history, including columns defined
                      in `self.config` (e.g., date, event category, LoT date).
            'constant': pd.DataFrame with static patient information.
        max_nr_samples_per_split : int
            The maximum number of distinct event prediction tasks (different event
            categories or prediction windows) to generate for *each* selected split date.
            The actual number might be less due to random sampling and avoiding duplicates.
        max_num_splits_per_split_event: int, optional
            When split dates are *not* overridden, this determines the maximum number
            of random split dates to select per unique LoT start date during the
            initial candidate selection. Defaults to 1.
        reference_split_dates : pd.DataFrame, optional
            A DataFrame containing specific split dates to use, typically generated by
            another data splitter (e.g., DataSplitterForecasting) to ensure alignment
            between different task types. Must contain the columns specified in
            `self.config.date_col` and `self.config.split_date_col`. If provided,
            `override_split_dates` must be None. Defaults to None.
        override_split_dates : list, optional
            A list of specific datetime objects to use as split dates, typically for
            inference scenarios. If provided, `reference_split_dates` must be None.
            Defaults to None.
        override_category : str, optional
            If provided, forces the sampling process to only consider this specific
            event category for prediction, instead of randomly sampling from available
            categories. Defaults to None.
        override_observation_time_delta : pd.Timedelta, optional
            If provided, forces the prediction window to be exactly this duration,
            instead of randomly sampling a window duration. Defaults to None.

        Returns
        -------
        list[DataSplitterEventsGroup]
            A list where each element corresponds to one of the selected split dates.
            Each element is a DataSplitterEventsGroup containing multiple DataSplitterEventsOption objects.
            Each option represents a single event prediction task (split) and contains attributes as
            defined in DataSplitterEventsOption class.

        Raises
        ------
        ValueError
            If both `reference_split_dates` and `override_split_dates` are provided.
            If required columns are missing in `patient_data['events']`.
        AssertionError
            If internal checks fail, e.g., when using `reference_split_dates` and
            consistency checks with potential dates fail.
        TypeError
            If input arguments have incorrect types.
        """

        # --- Assertions ---

        # Input Type Assertions
        assert isinstance(patient_data, dict), "patient_data must be a dictionary."
        assert isinstance(max_nr_samples_per_split, int) and max_nr_samples_per_split > 0, "max_nr_samples_per_split "
        "must be a positive integer."
        assert isinstance(max_num_splits_per_split_event, int) and max_num_splits_per_split_event > 0, (
            "max_num_samples_per_lot must be a positive integer."
        )
        assert reference_split_dates is None or isinstance(reference_split_dates, pd.DataFrame), (
            "reference_split_dates must be None or a pandas DataFrame."
        )
        assert override_split_dates is None or isinstance(override_split_dates, list), (
            "override_split_dates must be None or a list."
        )
        assert override_category is None or isinstance(override_category, str), (
            "override_category must be None or a string."
        )
        assert override_observation_time_delta is None or isinstance(override_observation_time_delta, pd.Timedelta), (
            "override_observation_time_delta must be None or a pandas Timedelta."
        )

        # Input Data Structure and Content Assertions
        assert "events" in patient_data, "patient_data dictionary must contain the key 'events'."
        assert "constant" in patient_data, "patient_data dictionary must contain the key 'constant'."
        assert isinstance(patient_data["events"], pd.DataFrame), "patient_data['events'] must be a pandas DataFrame."
        assert isinstance(patient_data["constant"], pd.DataFrame), (
            "patient_data['constant'] must be a pandas DataFrame."
        )

        # Check for required columns in the events dataframe
        required_event_cols = [
            self.config.date_col,
            self.config.event_category_col,
            self.config.event_name_col,
        ]
        missing_cols = [col for col in required_event_cols if col not in patient_data["events"].columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in patient_data['events']: {missing_cols}")

        # Mutual Exclusivity Assertion for Split Date Sources
        assert reference_split_dates is None or override_split_dates is None, (
            "Cannot provide both reference_split_dates and override_split_dates."
        )

        #: get all possible splits
        events = patient_data["events"]
        events = events.sort_values(self.config.date_col)

        # Do some quick sanity checks
        if self.config.warning_for_splitters_patient_without_lots:
            lot_events = events[events[self.config.event_category_col] == self.config.event_category_lot]
            if lot_events.shape[0] == 0:
                logging.warning(
                    "Patient "
                    + str(patient_data["constant"][self.config.patient_id_col].iloc[0])
                    + " has no LoT events. Forecasting splits may be invalid."
                    "To disable this warning, set warning_for_splitters_patient_without_lots to False in config."
                )

        if reference_split_dates is None and override_split_dates is None:
            #: get unique dates, if needed
            pot_all_possible_split_dates = self._get_all_dates_within_range_of_split_event(
                patient_data, pd.Timedelta(0), self.max_split_length_after_split_event
            )
            pot_all_possible_split_dates = self.select_random_splits(
                pot_all_possible_split_dates,
                max_num_splits_per_split_event=max_num_splits_per_split_event,
            )

            all_possible_split_dates = pot_all_possible_split_dates

        elif reference_split_dates is not None:
            # Set to the preselected split dates, and do some assertions
            all_possible_split_dates = reference_split_dates.copy()
            all_possible_split_dates = all_possible_split_dates.reset_index(drop=True)
            assert all_possible_split_dates[self.config.date_col].isna().sum() == 0, "Still missing dates"

        elif override_split_dates is not None:
            # If we're overriding the split dates, then we need to create a new dataframe
            all_possible_split_dates = pd.DataFrame(
                {
                    self.config.date_col: override_split_dates,
                    self.config.split_date_col: [pd.NA] * len(override_split_dates),
                }
            )

        else:
            raise ValueError("Invalid split dates provided")

        ret_splits = []

        for curr_sample_index in range(len(all_possible_split_dates)):
            #: get current data
            curr_date, lot_date = all_possible_split_dates.iloc[curr_sample_index, :].tolist()

            #: get the input & output data
            events_before_split = events[events[self.config.date_col] <= curr_date]
            events_after_split = events[events[self.config.date_col] > curr_date]

            prev_sampled_category = []
            ret_split_lot = DataSplitterEventsGroup()

            #: loop through 1 to max_nr_samples_per_split
            for _ in range(max_nr_samples_per_split):
                #: sample variables
                sampled_cateogry, sampled_var_name = self._sample_manual_variables(
                    events_after_split, override_category
                )

                #: check if we sampled the same category as before
                if sampled_cateogry in prev_sampled_category:
                    continue
                prev_sampled_category.append(sampled_cateogry)

                # Determine how many weeks to predict into the future
                if override_observation_time_delta is None:
                    #: randomly sample end date -> so that we also get random values in between for consistency
                    # This is so that the model can learn different time values for the same variable
                    #: To not bias the model, we select a random nr time as max end date``

                    if self.unit_length_to_sample == "days":
                        max_units = self.max_length_to_sample.days
                        min_units = self.min_length_to_sample.days
                        random_units = np.random.randint(min_units, max_units + 1)
                        end_time_delta = pd.Timedelta(days=random_units)
                    elif self.unit_length_to_sample == "weeks":
                        max_units = self.max_length_to_sample.days // 7
                        min_units = self.min_length_to_sample.days // 7
                        random_units = np.random.randint(min_units, max_units + 1)
                        end_time_delta = pd.Timedelta(weeks=random_units)
                    else:
                        raise NotImplementedError(
                            f"Unit length to sample {self.unit_length_to_sample} not implemented."
                        )
                else:
                    end_time_delta = override_observation_time_delta

                # Process the actual end date
                end_date = curr_date + end_time_delta
                end_date = max(end_date, events_after_split[self.config.date_col].min())
                end_date_within_data = end_date <= events[self.config.date_col].max()
                events_limited_after_split = events_after_split[events_after_split[self.config.date_col] <= end_date]

                # Get the events
                diagnosis_after_split = events_limited_after_split[
                    events_limited_after_split[self.config.event_category_col] == sampled_cateogry
                ]
                lot_after_split = events_limited_after_split[
                    events_limited_after_split[self.config.event_category_col] == self.config.event_category_lot
                ]
                death_after_split = events_limited_after_split[
                    events_limited_after_split[self.config.event_name_col] == self.config.event_category_death
                ]

                #: apply censoring using next_lot_date
                next_lot_date = lot_after_split[self.config.date_col].min() if len(lot_after_split) > 0 else None
                next_death_date = death_after_split[self.config.date_col].min() if len(death_after_split) > 0 else None

                #: determine whether occurred, censored & if so, which date
                occurred = None
                censored = None
                date_occurred = end_date

                if diagnosis_after_split.shape[0] > 0:
                    # Event occurred within end date
                    occurred = True

                    # If an lot occurred first though, then we're censored
                    if next_lot_date is not None and diagnosis_after_split[self.config.date_col].min() > next_lot_date:
                        censored = "new_therapy_start"
                        occurred = False

                else:
                    # Event did not occur
                    occurred = False

                    if next_lot_date is not None:
                        # If we were censored by the next lot date
                        censored = "new_therapy_start"

                    elif next_death_date is not None:
                        # If death occurred then not censored, since this is the only time we
                        # actually know event didn't occur
                        # In case we're sampling for death var, and it occurred, then it wouldn't trigger this logic
                        censored = None

                    elif end_date_within_data:
                        # Event did not occur within the given time frame
                        censored = None

                    else:
                        # If we were censored by the end of the data, but not death
                        censored = "end_of_data"

                # Check for data cutoff
                if self.config.date_cutoff is not None:
                    if censored is None and occurred is False and end_date > self.config.date_cutoff:
                        # Check if outside of date cutoff
                        # if occurred is False and not censored, then we know event didn't occur in the mean time
                        occurred = False
                        censored = "data_cutoff"

                #: add to return list
                ret_split_lot.append(
                    DataSplitterEventsOption(
                        events_until_split=events_before_split,
                        constant_data=patient_data["constant"].copy(),
                        event_occurred=occurred,
                        event_censored=censored,
                        observation_end_date=date_occurred,
                        split_date_included_in_input=curr_date,
                        sampled_category=str(sampled_cateogry),
                        sampled_category_name=sampled_var_name,
                        lot_date=lot_date,
                    )
                )

            # Add for current LoT possible splits
            ret_splits.append(ret_split_lot)

        #: return list
        return ret_splits
