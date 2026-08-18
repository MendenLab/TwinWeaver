import logging

import pandas as pd

from twinweaver.common.data_manager import DataManager
from twinweaver.common.config import Config


class BaseDataSplitter:
    """
    Base splitter class, used for both time to event splitting and forecasting splitting.
    It implements some basic functionality that is shared between the two types of splitters.

    Attributes
    ----------
    no_split_before_events : list[str] | None
        Optional list of "gate" events before which no split may happen. Each entry is matched
        against both `config.event_category_col` and `config.event_name_col`, so either an event
        category (e.g. ``["lot"]``) or a specific event name can be given. Split dates are then
        restricted to be on or after the earliest matching event of the patient. Patients without
        any matching event yield no splits at all. Default: None (no gating).
    random_state : int | np.random.Generator | np.random.RandomState | None
        Random state used when sampling split dates in `select_random_splits`. When None (the
        default) the global NumPy random stream is used, which `Config` seeds from `Config.seed`.
    """

    def __init__(
        self,
        data_manager: DataManager,
        config: Config,
        max_split_length_after_split_event: pd.Timedelta = pd.Timedelta(days=90),
        no_split_before_events: list = None,
        random_state=None,
    ):
        """
        Constructor for the BaseDataSplitter class.

        Parameters
        ----------
        data_manager: DataManager
            the data manager object that holds the data.
        config: Config
            Configuration object holding constants.
        max_split_length_after_split_event: pd.Timedelta
            the maximum number of days after a LoT event that we want to consider as
            a starting point.
        no_split_before_events: list[str], optional
            Optional list of gate events before which no split may happen. Entries are matched
            against both the event category and the event name column, so e.g. ``["lot"]``
            (a category) or ``["treatment_start"]`` (an event name) both work. Splits are only
            allowed on or after the earliest matching event; patients without any matching event
            produce no splits. Defaults to None (no gating).
        random_state: int | np.random.Generator | np.random.RandomState, optional
            Random state used for sampling split dates. Defaults to None, which uses the global
            NumPy random stream (seeded from `Config.seed`).
        """

        assert config.split_event_category is not None, "config.split_event_category must be set (e.g. ['lab'])."

        if no_split_before_events is not None:
            assert isinstance(no_split_before_events, (list, tuple, set)), (
                "no_split_before_events must be a list of strings (event categories and/or event names), "
                f"got {type(no_split_before_events).__name__}."
            )
            no_split_before_events = list(no_split_before_events)
            assert len(no_split_before_events) > 0, "no_split_before_events must not be empty (use None to disable)."
            assert all(isinstance(event, str) for event in no_split_before_events), (
                "All entries of no_split_before_events must be strings."
            )

        self.dm = data_manager
        self.config = config
        self.max_split_length_after_split_event = max_split_length_after_split_event
        self.no_split_before_events = no_split_before_events
        self.random_state = random_state

    def _get_earliest_allowed_split_date(self, events: pd.DataFrame):
        """
        Get the earliest date at which a split is allowed, based on `no_split_before_events`.

        Each entry of `no_split_before_events` is matched against both the event category and the
        event name column, so gate events can be given either as categories or as specific names.

        Parameters
        ----------
        events: pd.DataFrame
            The (unfiltered) events of a single patient.

        Returns
        -------
        pd.Timestamp | None
            The earliest date of any matching gate event, or None when gating is disabled.
            Returns pd.NaT when gating is enabled but the patient has no matching event, which
            means that no split is allowed for this patient at all.
        """

        if not self.no_split_before_events:
            return None

        gate_mask = events[self.config.event_category_col].isin(self.no_split_before_events) | events[
            self.config.event_name_col
        ].isin(self.no_split_before_events)
        gate_dates = events.loc[gate_mask, self.config.date_col]

        if gate_dates.shape[0] == 0:
            return pd.NaT

        return gate_dates.min()

    def _get_all_dates_within_range_of_split_event(
        self,
        patient_data_dic: dict,
        time_before_lot_start: pd.Timedelta,
        max_split_length_after_split_event: pd.Timedelta,
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

        #: if requested, do not allow any split before the gate event (e.g. treatment start).
        # Computed on the unfiltered events, so that a gate event which is also the last date
        # (or a death event) still gates correctly.
        earliest_allowed_split_date = self._get_earliest_allowed_split_date(events)
        if earliest_allowed_split_date is not None:
            if pd.isna(earliest_allowed_split_date):
                #: patient has none of the gate events -> no splits allowed at all
                return pd.DataFrame(columns=[self.config.date_col, self.config.split_date_col])
            events = events[events[self.config.date_col] >= earliest_allowed_split_date]

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

        #: restrict search space to only events that are within max_split_length_after_split_event days after LoT
        all_possible_dates = []
        for curr_split_date in all_split_dates:
            for curr_date in all_dates:
                if (
                    curr_date <= curr_split_date + max_split_length_after_split_event
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

    def select_random_splits(
        self, all_possible_split_dates: pd.DataFrame, max_num_splits_per_split_event: int = 1
    ) -> pd.DataFrame:
        """
        Select random splits within a given lot, based on the input split dates.
        Thus each LoT has up to max_num_splits_per_split_event random splits.

        Sampling is done *without* replacement, so the same date is never returned twice for the
        same split event. When a split event has fewer candidate dates than requested, all of its
        candidates are returned (i.e. the number of splits is capped, not padded with duplicates).

        The random stream is `self.random_state`; when that is None (default) the global NumPy
        stream is used, which `Config` seeds from `Config.seed`.

        Parameters
        ----------
        all_possible_split_dates: pd.DataFrame
            a pandas dataframe that holds all possible split dates for the given patient data dictionary.
            It has columns self.config.date_col, self.config.split_date_col.
            Each row is a date which is valid for a split.

        max_num_splits_per_split_event: int
            the maximum number of samples per lot that we want to sample.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe that holds up to max_num_splits_per_split_event randomly selected
            split dates for each unique lot date.
            It has columns self.config.date_col, self.config.split_date_col.
            Each row is a date which is valid for a split.
        """

        # Edge case handling - nothing to sample from
        if all_possible_split_dates.shape[0] == 0:
            return all_possible_split_dates

        #: shuffle all candidates, then keep the first max_num_splits_per_split_event per split event.
        # This samples without replacement and naturally caps at the number of available candidates.
        shuffled = all_possible_split_dates.sample(frac=1, random_state=self.random_state)
        randomly_selected_per_lot = (
            shuffled.groupby(self.config.split_date_col, sort=False)
            .head(max_num_splits_per_split_event)
            .reset_index(drop=True)
        )

        #: inform the user when a split event had fewer candidate dates than requested
        nr_split_events = all_possible_split_dates[self.config.split_date_col].nunique(dropna=False)
        if randomly_selected_per_lot.shape[0] < nr_split_events * max_num_splits_per_split_event:
            logging.info(
                "Fewer candidate split dates than requested for at least one split event: "
                f"got {randomly_selected_per_lot.shape[0]} split dates for {nr_split_events} split event(s) "
                f"with max_num_splits_per_split_event={max_num_splits_per_split_event}."
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
