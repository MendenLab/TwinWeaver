import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def aggregate_events_to_weeks(
    df: pd.DataFrame,
    patientid_column: str = "patientid",
    date_column: str = "date",
    event_name_column: str = "event_name",
    event_value_column: str = "event_value",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregates a long-format events DataFrame to rounded weeks relative to each patient's first visit.

    This function rounds event dates to the nearest week (relative to each patient's
    first visit date), then aggregates multiple events that fall within the same week.
    For identical events (same `event_name`) within the same week:
    - Numerical values are averaged.
    - Categorical values use the mode (most frequent value), with random selection
      as a tiebreaker if multiple modes exist.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing longitudinal patient events in TwinWeaver format.
        Expected columns: patientid, date, event_category, event_name, event_value,
        event_descriptive_name, meta_data, source.
    patientid_column : str, default "patientid"
        The name of the column containing patient identifiers.
    date_column : str, default "date"
        The name of the column containing date/time information.
    event_name_column : str, default "event_name"
        The name of the column containing event names (used to identify identical events).
    event_value_column : str, default "event_value"
        The name of the column containing event values to aggregate.
    random_state : int, optional
        Random seed for reproducibility when breaking ties in mode selection.
        If None, results may vary for tied modes.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with dates rounded to weeks and events aggregated.
        The date column will contain dates representing the start of each week
        relative to the patient's first visit.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'patientid': ['p1', 'p1', 'p1', 'p1', 'p2', 'p2'],
    ...     'date': ['2024-01-01', '2024-01-03', '2024-01-08', '2024-01-09',
    ...              '2024-02-01', '2024-02-05'],
    ...     'event_category': ['lab', 'lab', 'lab', 'lab', 'lab', 'lab'],
    ...     'event_name': ['glucose', 'glucose', 'glucose', 'glucose', 'glucose', 'glucose'],
    ...     'event_value': [100, 110, 120, 130, 90, 95],
    ...     'event_descriptive_name': ['Glucose', 'Glucose', 'Glucose', 'Glucose',
    ...                                 'Glucose', 'Glucose'],
    ...     'meta_data': [None] * 6,
    ...     'source': ['events'] * 6,
    ... }
    >>> df = pd.DataFrame(data)
    >>> df['date'] = pd.to_datetime(df['date'])
    >>> result = aggregate_events_to_weeks(df)
    >>> # p1: Jan 1 and Jan 3 are in week 0 -> averaged to 105
    >>> #     Jan 8 and Jan 9 are in week 1 -> averaged to 125
    >>> # p2: Feb 1 and Feb 5 are in week 0 -> averaged to 92.5

    Notes
    -----
    - Weeks are calculated as 7-day intervals from each patient's first visit.
    - A date is assigned to week N if it falls within [first_visit + N*7 days,
      first_visit + (N+1)*7 days).
    - The output date for each week is the first day of that week interval.
    - Non-grouping columns (like event_descriptive_name, meta_data, source) take
      the first value within each aggregation group.
    - Empty DataFrames are returned as-is.

    """
    if df.empty:
        return df.copy()

    # Validate input columns exist
    required_columns = [patientid_column, date_column, event_name_column, event_value_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Calculate the first visit date for each patient
    first_visits = df.groupby(patientid_column)[date_column].min().reset_index()
    first_visits.columns = [patientid_column, "_first_visit_date"]

    # Merge first visit dates
    df = df.merge(first_visits, on=patientid_column, how="left")

    # Calculate days since first visit and round to weeks
    df["_days_since_first"] = (df[date_column] - df["_first_visit_date"]).dt.days
    df["_week_number"] = (df["_days_since_first"] / 7).apply(np.floor).astype(int)

    # Calculate the rounded date (start of the week)
    df["_rounded_date"] = df["_first_visit_date"] + pd.to_timedelta(df["_week_number"] * 7, unit="D")

    # Identify grouping columns
    grouping_cols = [patientid_column, "_rounded_date", event_name_column]

    # Identify other columns in the dataframe
    all_columns = df.columns.tolist()
    temp_columns = ["_first_visit_date", "_days_since_first", "_week_number", "_rounded_date"]
    other_columns = [
        col for col in all_columns if col not in grouping_cols + [date_column, event_value_column] + temp_columns
    ]

    def aggregate_values(group: pd.DataFrame) -> pd.Series:
        """Aggregate event values within a group."""
        values = group[event_value_column]

        # Try to convert to numeric
        numeric_values = pd.to_numeric(values, errors="coerce")

        if numeric_values.notna().all() and len(numeric_values) > 0:
            # All values are numeric - use mean
            aggregated_value = numeric_values.mean()
        else:
            # Categorical - use mode
            mode_result = values.mode()
            if len(mode_result) == 0:
                # All NaN
                aggregated_value = values.iloc[0] if len(values) > 0 else np.nan
            elif len(mode_result) == 1:
                aggregated_value = mode_result.iloc[0]
            else:
                # Multiple modes - random selection
                aggregated_value = np.random.choice(mode_result.values)

        # Build result series - grouping columns are available in the group
        result = {
            event_value_column: aggregated_value,
        }

        # Take first value for other columns (non-grouping, non-date, non-temp columns)
        for col in other_columns:
            if col in group.columns:
                result[col] = group[col].iloc[0]

        return pd.Series(result)

    # Group and aggregate
    aggregated = df.groupby(grouping_cols, as_index=False).apply(aggregate_values, include_groups=False)

    # The groupby with as_index=False returns the grouping columns, rename _rounded_date to date_column
    aggregated = aggregated.rename(columns={"_rounded_date": date_column})

    # Ensure proper column order (original order where possible)
    original_cols = [col for col in df.columns if col not in temp_columns]
    final_cols = [col for col in original_cols if col in aggregated.columns]
    aggregated = aggregated[final_cols]

    # Sort by patient and date
    aggregated = aggregated.sort_values([patientid_column, date_column]).reset_index(drop=True)

    return aggregated


def identify_constant_and_changing_columns(
    df: pd.DataFrame,
    date_column: str,
    patientid_column: str,
) -> Tuple[List[str], List[str]]:
    """Identifies which columns remain constant and which change over time for each patient.

    This function analyzes a DataFrame to determine which columns have values that
    stay constant across all time points for each patient, and which columns have
    values that change over time for at least one patient.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing patient data with multiple time points.
    date_column : str
        The name of the column containing date/time information.
    patientid_column : str
        The name of the column containing patient identifiers.

    Returns
    -------
    constant_columns : list of str
        A list of column names that remain constant across all time points
        for every patient.
    changing_columns : list of str
        A list of column names that change over time for at least one patient.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'patient_id': [1, 1, 2, 2],
    ...     'date': ['2024-01-01', '2024-02-01', '2024-01-01', '2024-02-01'],
    ...     'age': [30, 30, 45, 45],
    ...     'weight': [70, 72, 80, 80],
    ...     'gender': ['M', 'M', 'F', 'F']
    ... }
    >>> df = pd.DataFrame(data)
    >>> constant, changing = identify_constant_and_changing_columns(
    ...     df, date_column='date', patientid_column='patient_id'
    ... )
    >>> print(constant)
    ['age', 'gender']
    >>> print(changing)
    ['weight']

    Notes
    -----
    - The date_column and patientid_column are excluded from the analysis.
    - A column is considered constant if all values for a patient are identical
      (including NaN values, which are treated as equal to each other).
    - A column is considered changing if at least one patient has different
      values across their time points.

    """
    # Validate input columns exist
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    if patientid_column not in df.columns:
        raise ValueError(f"Patient ID column '{patientid_column}' not found in DataFrame")

    # Get columns to analyze (exclude date and patientid columns)
    columns_to_analyze = [col for col in df.columns if col not in [date_column, patientid_column]]

    constant_columns = []
    changing_columns = []

    for col in columns_to_analyze:
        # Group by patient and check if values are constant within each patient
        # Use nunique with dropna=False to count NaN as a distinct value
        unique_values_per_patient = df.groupby(patientid_column)[col].nunique(dropna=False)

        # A column changes if any patient has more than one unique value
        if (unique_values_per_patient > 1).any():
            changing_columns.append(col)
        else:
            constant_columns.append(col)

    return constant_columns, changing_columns
