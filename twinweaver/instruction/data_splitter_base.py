import pandas as pd

from twinweaver.common.data_manager import DataManager
from twinweaver.common.config import Config


class BaseDataSplitter:
    """
    Base splitter class, used for both time to event splitting and forecasting splitting.
    It implements some basic functionality that is shared between the two types of splitters.
    """

    def __init__(
        self,
        data_manager: DataManager,
        config: Config,
        max_split_length_after_lot: pd.Timedelta = pd.Timedelta(days=90),
        max_lookback_time_for_value: pd.Timedelta = pd.Timedelta(days=90),
        max_forecast_time_for_value: pd.Timedelta = pd.Timedelta(days=90),
    ):
        """
        Constructor for the BaseDataSplitter class.

        Parameters
        ----------
        data_manager: DataManager
            the data manager object that holds the data.
        config: Config
            Configuration object holding constants.
        max_split_length_after_lot: pd.Timedelta
            the maximum number of days after a LoT event that we want to consider as
            a starting point.
        max_lookback_time_for_value: pd.Timedelta
            the maximum number of days before a certain split date where we need to see
            the value of the target variable.
        max_forecast_time_for_value : pd.Timedelta
            the maximum number of days after a certain split date where we need to see
            the value of the target variable when filtering.
        """

        self.dm = data_manager
        self.config = config
        self.max_split_length_after_lot = max_split_length_after_lot
        self.max_lookback_time_for_value = max_lookback_time_for_value
        self.max_forecast_time_for_value = max_forecast_time_for_value

    def _get_all_dates_within_range_of_lot(
        self,
        patient_data_dic: dict,
        time_before_lot_start: pd.Timedelta,
        max_split_length_after_lot: pd.Timedelta,
    ) -> pd.DataFrame:
        """
        Get all possible valid split dates for a given patient data dictionary, without
        any filtering regarding variables, used in some helper functions

        Parameters
        ----------
        patient_data_dic: dict
            the patient data dictionary that holds the data for a given patient.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe that holds all possible split dates for the given patient data dictionary.
            It has columns self.config.date_col and self.config.split_date_col.
            Each row is a date which is valid for a split.
        """

        #: setup data
        events = patient_data_dic["events"].copy()

        if self.config.event_category_death in events[self.config.event_category_col].unique():
            # Exclude death events for splitting, to avoid edge cases
            events = events[events[self.config.event_category_col] != self.config.event_category_death]
        else:
            # Exclude last date for splitting
            events = events[events[self.config.date_col] != events[self.config.date_col].max()]

        #: get all starting split dates, sorted, excluding death
        all_split_dates = events[events[self.config.event_category_col] == self.config.split_event_category][
            self.config.date_col
        ]
        all_split_dates = list(set(all_split_dates.tolist()))
        all_split_dates.sort()

        #: go over all events
        all_dates = events[self.config.date_col].copy()
        all_dates = list(set(all_dates.tolist()))
        all_dates.sort()

        #: restrict search space to only events that are within max_split_length_after_lot days after LoT
        all_possible_dates = []
        for curr_split_date in all_split_dates:
            for curr_date in all_dates:
                if (
                    curr_date <= curr_split_date + max_split_length_after_lot
                    and curr_date >= curr_split_date - time_before_lot_start
                ):
                    all_possible_dates.append((curr_date, curr_split_date))

        # Serve as df
        df = pd.DataFrame(all_possible_dates, columns=[self.config.date_col, self.config.split_date_col])
        if df.shape[0] == 0:
            return df

        #: keep only unique dates, using the one with closest split_date
        df["diff"] = (df[self.config.date_col] - df[self.config.split_date_col]).dt.days
        df["diff_abs"] = df["diff"].abs()
        df = df.loc[df.groupby(self.config.date_col)["diff_abs"].idxmin()]
        df = df.drop(columns=["diff", "diff_abs"])

        return df

    def select_random_splits_within_lot(
        self, all_possible_split_dates: pd.DataFrame, max_num_splits_per_lot: int = 1
    ) -> pd.DataFrame:
        """
        Select random splits within a given lot, based on the input split dates.
        Thus each LoT has max_num_splits_per_lot random split.


        Parameters
        ----------
        all_possible_split_dates: pd.DataFrame
            a pandas dataframe that holds all possible split dates for the given patient data dictionary.
            It has columns self.config.date_col, self.config.split_date_col.
            Each row is a date which is valid for a split.

        max_num_splits_per_lot: int
            the maximum number of samples per lot that we want to sample.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe that holds a randomly selected split date for each unique lot date.
            It has columns self.config.date_col, self.config.split_date_col.
            Each row is a date which is valid for a split.
        """

        #: select one randomly per unique LOT_self.config.date_col
        randomly_selected_per_lot = (
            all_possible_split_dates.groupby(self.config.split_date_col)
            .sample(n=max_num_splits_per_lot, replace=True, random_state=1)
            .reset_index(drop=True)
        )

        # Sort
        randomly_selected_per_lot = randomly_selected_per_lot.sort_values(
            by=[self.config.split_date_col, self.config.date_col]
        )

        #: return
        return randomly_selected_per_lot

    def drop_duplicates_except_na_for_date_col(self, df):
        """
        Drops duplicates from the DataFrame except for rows with NA in the date column.
        Note: Original function description mentioned split_date_col, but implementation uses date_col.
              Following the implementation.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with duplicates dropped, except for rows with NA in the date column.
        """

        # Edge case handling
        if df.shape[0] == 0:
            return df

        # Split the DataFrame into rows with NA in the specified column and rows without NA
        na_rows = df[df[self.config.date_col].isna()]
        non_na_rows = df[~df[self.config.date_col].isna()]

        # Drop duplicates from the rows without NA
        non_na_rows_deduped = non_na_rows.drop_duplicates()

        # Concatenate the NA rows and the deduplicated non-NA rows
        result_df = pd.concat([na_rows, non_na_rows_deduped])

        # Sort by index
        result_df = result_df.sort_index()

        return result_df
