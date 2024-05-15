import pandas as pd
import logging
from typing import List
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder


def read_data(file_path, dtype=None, **kwargs) -> pd.DataFrame:
    """
    Reads data from a file using pandas and returns a DataFrame.

    Args:
        file_path (str): The path to the file.
        dtype (dict or None, optional): Data type specification for columns.
        Defaults to None.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv().

    Returns:
        pandas.DataFrame: The DataFrame containing the read data.
    """
    try:
        df = pd.read_csv(file_path, dtype=dtype, **kwargs)
        return df
    except Exception as e:
        print(f"Error reading data from {file_path}: {e}")
        return None


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drops specified columns from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame from which columns
        are to be dropped.
        columns_to_drop (list): List of column names to be dropped.

    Returns:
        pandas.DataFrame: The DataFrame with specified columns dropped.
    """
    try:
        df = df.drop(columns=columns_to_drop, axis=1)
        logging.info("Columns dropped successfully")
        return df
    except Exception as e:
        logging.error(f"Error dropping columns: {e}")
        return None


def get_high_cardinality_columns(df: pd.DataFrame, threshold: int):
    """
    Get columns with high cardinality from a pandas DataFrame
    based on a threshold.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        threshold (int): The threshold value to determine high cardinality.

    Returns:
        list: A list of column names with high cardinality.
    """
    try:
        categorical_columns = [
            col for col in df.columns if df[col].dtype == object
        ]
        high_cardinality_columns = [
            col for col in categorical_columns if df[col].nunique() > threshold
        ]
        return high_cardinality_columns
    except Exception as e:
        logging.error(f"Error getting high cardinality columns: {e}")
        return None


def find_columns_with_missing_values(data: pd.DataFrame, threshold: float):
    """
    Find columns in the dataset with more than a specified percentage of
    missing values.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    threshold (float): The percentage threshold for considering a column as
    having too many missing values.

    Returns:
    list: A list of column names with more than the specified percentage of
    missing values.
    """
    total_rows = len(data)
    missing_value_threshold = threshold * total_rows

    columns_with_missing_values = []
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > missing_value_threshold:
            columns_with_missing_values.append(col)

    return columns_with_missing_values


def get_constant_variance_columns(data, threshold=0):
    """
    Calculate the columns with constant variance using VarianceThreshold.

    Parameters:
    - data: pandas DataFrame or array-like of shape (n_samples, n_features)
            Input data.
    - threshold: float, optional (default=0)
            Features with a variance lower than this threshold will be removed.

    Returns:
    - constant_variance_columns: list
            List of column names with constant variance.
    """
    # Initialize VarianceThreshold
    var_model = VarianceThreshold(threshold=threshold)

    # Fit to data
    var_model.fit(data)

    # Get boolean mask of selected features
    constant_features_mask = var_model.get_support()

    # Get column names with constant variance
    constant_variance_columns = data.columns[~constant_features_mask].tolist()

    return constant_variance_columns


def ordinal_encode_categorical_columns(data):
    """
    Perform ordinal encoding on categorical columns using OrdinalEncoder.

    Parameters:
    - data: pandas DataFrame
            Input DataFrame containing categorical columns to be encoded.

    Returns:
    - encoded_data: pandas DataFrame
            DataFrame with categorical columns encoded using ordinal encoding.
    - encoder: OrdinalEncoder
            Fitted OrdinalEncoder object used for encoding.
    """

    # Selecting categorical columns
    categorical_columns = data.select_dtypes(
        include=['object']).columns.tolist()

    # Creating DataFrame with only categorical columns
    categorical_data = data[categorical_columns]

    # Initialize OrdinalEncoder
    ord_enc = OrdinalEncoder()

    # Fit and transform the data using the encoder
    data_encoded = ord_enc.fit_transform(categorical_data)

    # Convert the encoded data back to a DataFrame
    categorical_data_encoded = pd.DataFrame(data_encoded,
                                            columns=categorical_data.columns)

    return categorical_data_encoded


def find_cat_and_num_columns(data):
    """
    Find categorical and numerical columns in a DataFrame.

    Parameters:
    - data: pandas DataFrame
            Input DataFrame.

    Returns:
    - categorical_columns: list
            List of categorical column names.
    - numerical_columns: list
            List of numerical column names.
    """

    # Selecting categorical columns
    categorical_columns = data.select_dtypes(
        include=['object']).columns.tolist()

    # Selecting numerical columns
    numerical_columns = data.select_dtypes(
        include=['number']).columns.tolist()

    return categorical_columns, numerical_columns
