import random
import pandas as pd
import numpy as np
import logging
from twinweaver.common.config import (
    Config,
)


class DataManager:
    """
    Manages data loading, processing, and splitting for a single indication.

    This class handles the lifecycle of data for one specific indication,
    including loading data from files (or using overridden dataframes),
    performing processing steps like date conversion and cleaning, ensuring
    unique event naming, and splitting the patient data into training,
    validation, and test sets based on patient IDs. It utilizes a `Config`
    object for various settings and column names.
    """

    def __init__(
        self,
        config: Config,  # Added config parameter
        train_split_min: float = 0.8,
        validation_split_max: float = 0.1,
        test_split_max: float = 0.1,
        max_val_test_nr_patients: int = 500,
        replace_special_symbols_override: list = None,
    ) -> None:
        """
        Initializes the DataManager for a specific indication.

        Sets up the manager with the configuration, data split
        parameters, and options for handling special characters in event names.

        Parameters
        ----------
        config : Config
            A configuration object containing paths, column names, category names,
            and other constants used throughout the data management process.
        train_split_min : float, optional
            The minimum proportion of patients to allocate to the training set.
            Defaults to 0.8. The actual number will be the remainder after
            allocating validation and test sets.
        validation_split_max : float, optional
            The maximum proportion of the total patients to allocate to the
            validation set. The actual number is capped by
            `max_val_test_nr_patients`. Defaults to 0.1.
        test_split_max : float, optional
            The maximum proportion of the total patients to allocate to the
            test set. The actual number is capped by `max_val_test_nr_patients`.
            Defaults to 0.1.
        max_val_test_nr_patients : int, optional
            The absolute maximum number of patients to include in the validation
            and test sets combined. Defaults to 500.
        replace_special_symbols_override : list, optional
            A list of tuples to override the default special character replacements
            in event descriptive names. Each tuple should be in the format
            `(event_category, (string_to_replace, replacement_string))`. If None,
            default replacements specified in the method are used. Defaults to None.
        """

        #: initialize the data manager
        self.config = config  # Store config object
        self.train_split = train_split_min
        self.validation_split = validation_split_max
        self.test_split = test_split_max
        self.max_val_test_nr_patients = max_val_test_nr_patients
        self.variable_types = {}  # event_name -> "numeric" / "categorical"

        # Setup replacing of special symbol, format is event_category : (<string_to_replace>, <replacement_string>)
        if replace_special_symbols_override is not None:
            self.replace_special_symbols = replace_special_symbols_override
        else:
            # Use config constants for event categories where available
            self.replace_special_symbols = [
                (self.config.event_category_labs, ("/", " per ")),
                (self.config.event_category_labs, (".", " ")),
                (
                    "drug",
                    ("/", " "),
                ),  # "drug" category not explicitly in Config constants provided
                (
                    self.config.event_category_lot,
                    ("/", " "),
                ),  # Use config for 'lot' category
            ]

        # Setup indication
        self.data_frames = None
        self.unique_events = None
        self.all_patientids = None

        # Set seed
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

    def load_indication_data(
        self, df_events: pd.DataFrame, df_constant: pd.DataFrame, df_constant_description: pd.DataFrame
    ) -> None:
        """
        Loads the data tables (as dataframes) for the specified indication.
        It also removes any columns named "Unnamed: *" from the loaded DataFrames.

        Parameters
        ----------
        df_events : pd.DataFrame
            The events dataframe containing time-series data.
        df_constant : pd.DataFrame
            The constant dataframe containing static patient data.
        df_constant_description : pd.DataFrame
            The dataframe describing the constant variables.
        """

        # Copy over
        df_events = df_events.copy()
        df_constant = df_constant.copy()
        df_constant_description = df_constant_description.copy()

        # Do some basic checks
        assert df_events.shape[0] > 0, "df_events is empty"
        assert df_constant.shape[0] > 0, "df_constant is empty"
        assert df_constant_description.shape[0] > 0, "df_constant_description is empty"

        # Assert cols in events
        assert self.config.patient_id_col in df_events.columns, (
            f"Patient ID column {self.config.patient_id_col} not in events dataframe"
        )
        assert self.config.event_descriptive_name_col in df_events.columns, (
            f"Event descriptive name column {self.config.event_descriptive_name_col} not in events dataframe"
        )
        assert self.config.event_value_col in df_events.columns, (
            f"Event value column {self.config.event_value_col} not in events dataframe"
        )
        assert self.config.date_col in df_events.columns, f"Date column {self.config.date_col} not in events dataframe"

        # Fil in missing columns

        # If no event category, set it to "unknown"
        if self.config.event_category_col not in df_events.columns:
            df_events[self.config.event_category_col] = self.config.event_category_default_value

        # If no event name, set it to event_descriptive_name
        if self.config.event_name_col not in df_events.columns:
            df_events[self.config.event_name_col] = df_events[self.config.event_descriptive_name_col]

        # If not meta column, set to empty
        if self.config.meta_data_col not in df_events.columns:
            df_events[self.config.meta_data_col] = self.config.event_meta_default_value

        # If no source columns, set it to "events"
        if self.config.source_col not in df_events.columns:
            df_events[self.config.source_col] = self.config.source_col_default_value

        # Assert cols in constant
        assert self.config.patient_id_col in df_constant.columns, (
            f"Patient ID column {self.config.patient_id_col} not in constant dataframe"
        )

        # assert cols in constant_description
        assert "variable" in df_constant_description.columns, "Column 'variable' not in constant_description dataframe"
        assert "comment" in df_constant_description.columns, "Column 'comment' not in constant_description dataframe"

        self.data_frames = {}
        self.data_frames["events"] = df_events
        self.data_frames["constant"] = df_constant
        self.data_frames["constant_description"] = df_constant_description

        #: drop all "Unnamed" columns
        def remove_unnamed_columns(df):
            return df.loc[:, ~df.columns.str.contains("^Unnamed")]

        for key in self.data_frames.keys():
            if self.data_frames[key] is not None:
                self.data_frames[key] = remove_unnamed_columns(self.data_frames[key])

        logging.info("Data loaded for indication")

    def process_indication_data(self) -> None:
        """
        Performs initial processing on the loaded indication data.

        Requires `load_indication_data` to be called first.
        This method converts the date columns (specified by `config.date_col`)
        in the 'events' DataFrame to datetime objects.
        It also checks for and removes rows with missing dates in these tables,
        logging a warning if any are found.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config.date_col and config.event_table_name
        events_table_key = self.config.event_table_name  # "events"
        date_col = self.config.date_col  # "date"

        #: convert for all COL_DATE column in each dataset to datetime
        self.data_frames[events_table_key][date_col] = pd.to_datetime(self.data_frames[events_table_key][date_col])

        # Check and drop all rows with missing date in events, and print warning if more than 0
        missing_date_events = self.data_frames[events_table_key][date_col].isnull().sum()
        total_events = len(self.data_frames[events_table_key])

        def handle_missing_dates(df_key, missing_count, total_count, col_date):
            if missing_count > 0:
                warning_msg = f"Found {missing_count} out of {total_count} missing dates in {df_key} "
                logging.warning(warning_msg)
                self.data_frames[df_key] = self.data_frames[df_key].dropna(subset=[col_date])

        # Use table keys and config.date_col
        handle_missing_dates(events_table_key, missing_date_events, total_events, date_col)

        logging.info("Data processed")

    def setup_unique_mapping_of_events(self) -> None:
        """
        Ensures uniqueness of descriptive event names and applies replacements.

        Requires `load_indication_data` to be called first.
        This method first identifies `event_descriptive_name` values that map to
        multiple `event_name` values within the same `event_category`. For these
        non-unique descriptive names, it appends the corresponding `event_name`
        to make them unique (e.g., "Measurement" becomes "Measurement - Systolic BP").

        Secondly, it applies predefined or overridden special character replacements
        (e.g., replacing "/" with " per " in lab results) to the
        `event_descriptive_name` column based on the `event_category`.

        Finally, it rebuilds the `self.unique_events` mapping (containing unique
        combinations of event_name, event_descriptive_name, and event_category)
        and asserts that all `event_descriptive_name` values are now unique.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        AssertionError
            If, after processing, the `event_descriptive_name` column still
            contains duplicate values.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config constants for column names
        event_name_col = self.config.event_name_col
        event_desc_name_col = self.config.event_descriptive_name_col
        event_cat_col = self.config.event_category_col
        events_table_key = self.config.event_table_name

        #: get all unique pairs of event_name and event_descriptive_name in self.data_frames["events"]
        self.unique_events = self.data_frames[events_table_key]
        self.unique_events = self.unique_events[[event_name_col, event_desc_name_col, event_cat_col]]
        self.unique_events = self.unique_events.copy().drop_duplicates()
        self.unique_events = self.unique_events.reset_index(drop=True)

        #: get all event_descriptive_name that are not unique
        non_unique_events = self.unique_events[event_desc_name_col].value_counts()
        non_unique_events = non_unique_events[non_unique_events > 1]

        # Extract corresponding event_name and event_category
        filtered_events = self.unique_events[event_desc_name_col]
        non_unique_events = self.unique_events[filtered_events.isin(non_unique_events.index)].copy()

        # create mapping for all non-unique descriptive names, and
        # then add event_name to those, and apply across entire dataset
        # Keep temporary column name as string literal
        non_unique_events["new_descriptive_name"] = (
            non_unique_events[event_desc_name_col] + " - " + non_unique_events[event_name_col]
        )
        # Use config constants for column names
        non_unique_events = non_unique_events[["new_descriptive_name", event_name_col, event_cat_col]]

        self.data_frames[events_table_key] = pd.merge(
            self.data_frames[events_table_key],
            non_unique_events,
            how="left",
            on=(event_name_col, event_cat_col),
        )  # Use config constants
        events_df = self.data_frames[events_table_key]
        new_desc_name = "new_descriptive_name"  # Keep temporary column name as string literal
        # Use config constant
        events_df[event_desc_name_col] = events_df[new_desc_name].fillna(events_df[event_desc_name_col])
        self.data_frames[events_table_key] = self.data_frames[events_table_key].drop(columns=["new_descriptive_name"])

        #: first convert special symbols in event_descriptive_name to alternatives, using self.replace_special_symbols
        for event_category, (
            string_to_replace,
            replacement_string,
        ) in self.replace_special_symbols:
            events_df = self.data_frames[events_table_key]
            # Use config constants
            category_mask = events_df[event_cat_col] == event_category
            desc_name_col = event_desc_name_col

            events_df.loc[category_mask, desc_name_col] = (
                events_df.loc[category_mask, desc_name_col]
                .astype(str)  # Ensure string type before replace
                .str.replace(
                    string_to_replace, replacement_string, regex=False
                )  # Added regex=False for literal replacement
            )

        #: recalculate self.unique_events and ensure no more non-unique event_descriptive_name
        # Use config constants
        cols_to_select = [event_name_col, event_desc_name_col, event_cat_col]
        self.unique_events = self.data_frames[events_table_key][cols_to_select].copy().drop_duplicates()
        self.unique_events = self.unique_events.reset_index(drop=True)

        # Assert that all unique now
        # Use config constant
        assert len(self.unique_events) == len(self.data_frames[events_table_key][event_desc_name_col].unique())

    def setup_dataset_splits(
        self,
    ) -> None:
        """
        Assigns each patient to a data split (train, validation, or test).

        Requires `load_indication_data` to be called first.
        The method determines the split assignment for each patient.
        It retrieves all unique patient IDs from the 'constant' data table.
        It calculates the number of patients for validation and test sets based on
        the `validation_split_max`, `test_split_max`, and `max_val_test_nr_patients`
        parameters set during initialization. The remaining patients are assigned to the training set #
        (calculated as the remainder after validation and test sets are allocated). Patients are randomly
        shuffled (with a fixed seed for reproducibility) before assignment.

        The resulting mapping (patient ID to split name) is assigned to the
        constant dataframe. It also stores all patient IDs in
        `self.all_patientids`. Asserts are performed to ensure the mapping covers
        all patients without overlap and that the split sizes match calculations.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        AssertionError
            If calculated splits do not match expected counts or if overlaps exist.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config constants
        patient_id_col = self.config.patient_id_col
        constant_table_key = "constant"  # Key remains "constant"
        train_split_name = self.config.train_split_name  # Use config for "train" split name

        # Raise warning if split column already exists in constant table
        if self.config.constant_split_col in self.data_frames[constant_table_key].columns:
            logging.warning(
                f"Column {self.config.constant_split_col} already exists in constant table. It will be overwritten."
            )

        #: get all patientids from self.data_frames["constant"]
        all_patients = self.data_frames[constant_table_key][patient_id_col].unique()
        self.all_patientids = all_patients

        #: get min(self.validation_split * num_patients, self.max_val_test_nr_patients)
        validation_nr_patients = min(
            int(self.validation_split * len(all_patients)),
            self.max_val_test_nr_patients,
        )

        #: then the same for test
        test_nr_patients = min(int(self.test_split * len(all_patients)), self.max_val_test_nr_patients)

        #: randomly shuffle with seed and split into train/val/test, using df.sample
        np.random.seed(self.config.seed)
        all_patients = np.random.permutation(all_patients)
        train_nr_patients = len(all_patients) - validation_nr_patients - test_nr_patients

        #: setup mapping so that each patientid returns which split it belongs to
        patient_to_split_mapping = {}
        # Use config.train_split_name for the train split key/value
        # Keep "validation" and "test" as strings since not defined in config
        patient_to_split_mapping.update({patient: train_split_name for patient in all_patients[:train_nr_patients]})
        patient_to_split_mapping.update(
            {
                patient: self.config.validation_split_name
                for patient in all_patients[train_nr_patients : train_nr_patients + validation_nr_patients]
            }
        )
        patient_to_split_mapping.update(
            {
                patient: self.config.test_split_name
                for patient in all_patients[train_nr_patients + validation_nr_patients :]
            }
        )

        #: assert that no overlap in patient mappings
        assert len(patient_to_split_mapping) == len(all_patients)

        #: assert that correct lengths
        # Use config.train_split_name for checking train split length
        assert (
            len([patient for patient, split in patient_to_split_mapping.items() if split == train_split_name])
            == train_nr_patients
        )
        assert (
            len(
                [
                    patient
                    for patient, split in patient_to_split_mapping.items()
                    if split == self.config.validation_split_name
                ]
            )
            == validation_nr_patients
        )
        assert (
            len(
                [patient for patient, split in patient_to_split_mapping.items() if split == self.config.test_split_name]
            )
            == test_nr_patients
        )

        # Assign to constant dataframe
        self.data_frames[constant_table_key][self.config.constant_split_col] = self.data_frames[constant_table_key][
            patient_id_col
        ].map(patient_to_split_mapping)

    def get_all_patientids_in_split(self, split: str) -> str:
        """
        Retrieves all patient IDs belonging to a specific data split.

        Parameters
        ----------
        split : str
            The name of the split (e.g., "train", "validation", "test").

        Returns
        -------
        list
            A list of patient ID strings belonging to the specified split.
        """
        # Use config constant for patient ID if needed, but here it's just a key lookup
        # patientid is the key itself.
        return (
            self.data_frames["constant"][self.data_frames["constant"][self.config.constant_split_col] == split][
                self.config.patient_id_col
            ]
            .unique()
            .tolist()
        )

    def get_patient_split(self, patientid: str) -> str:
        """
        Retrieves the split assignment for a specific patient.

        Parameters
        ----------
        patientid : str
            The unique identifier for the patient.

        Returns
        -------
        str
            The name of the split the patient belongs to.
        """
        return (
            self.data_frames["constant"]
            .loc[
                self.data_frames["constant"][self.config.patient_id_col] == patientid,
                self.config.constant_split_col,
            ]
            .values[0]
        )

    def get_patient_data(self, patientid: str) -> dict:
        """
        Retrieves and consolidates all data for a specific patient.

        Requires `load_indication_data` and `process_indication_data` to have
        been called. It's also recommended to call `setup_unique_mapping_of_events`
        to ensure consistent event naming.

        This method gathers data from the 'events', and 'constant'
        DataFrames for the specified `patientid`.
        - It filters the 'events' tables for the patient.
        - It filters the 'constant' table for the patient's static data.

        Parameters
        ----------
        patientid : str
            The unique identifier for the patient whose data is to be retrieved.

        Returns
        -------
        dict
            A dictionary containing the patient's data, with two keys:
            - "events": A pandas DataFrame containing all time-series events
                        (events data and sortedby date).
            - "constant": A pandas DataFrame containing the static (constant)
                          data for the patient.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        KeyError
            If essential columns specified in the config are missing from the
            dataframes after loading.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config constants for column names and table keys/sources where applicable
        patient_id_col = self.config.patient_id_col
        events_table_key = self.config.event_table_name  # "events"
        constant_table_key = "constant"  # Key remains "constant"

        #: get all data for a specific patient
        patient_data = {}

        #: first from events
        events = self.data_frames[events_table_key][
            self.data_frames[events_table_key][patient_id_col] == patientid
        ].copy()
        patient_data["events"] = events.sort_values(by=self.config.date_col)

        #: then from constant
        selected_patient = self.data_frames[constant_table_key][patient_id_col] == patientid
        patient_data["constant"] = self.data_frames[constant_table_key][selected_patient]

        # Remove any duplicates in case they get in events
        # Keep "events" key as string
        patient_data["events"] = patient_data["events"].drop_duplicates()

        #: return
        return patient_data

    def infer_var_types(self):
        """
        Fills self.dm.variable_types for every candidate forecasting variable.
        Classifies as "numeric" if at least `self.config.numeric_detect_min_fraction` of values
        can be parsed as numeric, otherwise "categorical".
        """

        events = self.data_frames[self.config.event_table_name]
        name_col = self.config.event_name_col
        value_col = self.config.event_value_col

        # Consider only variables in the configured forecasting categories
        mask_cat = events[self.config.event_category_col].isin(self.config.event_category_forecast)
        df = events.loc[mask_cat, [name_col, value_col]].copy()

        for var, sub in df.groupby(name_col):
            # Try numeric parse
            v = pd.to_numeric(sub[value_col], errors="coerce")
            frac_num = v.notna().mean()
            if frac_num >= self.config.numeric_detect_min_fraction:
                logging.info(f"Variable {var} classified as numeric ({frac_num:.2%} numeric values)")
                self.variable_types[var] = "numeric"
            else:
                logging.info(f"Variable {var} classified as categorical ({frac_num:.2%} numeric values)")
                self.variable_types[var] = "categorical"
