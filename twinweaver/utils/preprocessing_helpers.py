import pandas as pd
from typing import Tuple, List


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
