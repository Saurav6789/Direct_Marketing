from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import kendalltau


def select_k_best_features(data, target, k):
    """
    Perform chi-square test and select the K best features.

    Parameters:
    - data: pandas DataFrame
            Input DataFrame containing features.
    - target: pandas Series
            Target variable.
    - k: int
            Number of features to select.

    Returns:
    - selected_features: list
            Names of the selected features.
    """

    # Separate features and target variable
    X = data.drop(columns=[target])
    y = data[target]

    # Perform chi-square test
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_features = X.columns[selected_indices]

    return selected_features


def select_kendall_correlated_features(data, target, k):
    """
    Perform Kendall rank correlation between numerical columns and the target
    and select the top K features.

    Parameters:
    - data: pandas DataFrame
            Input DataFrame containing numerical features.
    - target: pandas Series
            Target variable.
    - k: int
            Number of features to select.

    Returns:
    - selected_features: list
            Names of the selected features.
    """

    # Separate features and target variable
    X = data.drop(columns=[target])  # Features
    y = data[target]  # Target variable

    # Calculate Kendall's tau coefficient for each numerical feature
    kendall_correlation = {}
    for column in X.columns:
        tau, _ = kendalltau(X[column], y)
        kendall_correlation[column] = abs(tau)

    # Sort features based on absolute Kendall's tau coefficient values
    sorted_features = sorted(kendall_correlation, key=kendall_correlation.get,
                             reverse=True)

    # Select the top K features
    selected_features = sorted_features[:k]

    return selected_features
