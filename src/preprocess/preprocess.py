from utility import read_data, drop_columns, get_high_cardinality_columns, \
    find_columns_with_missing_values, get_constant_variance_columns, \
    ordinal_encode_categorical_columns, find_cat_and_num_columns
from feature_selection import select_k_best_features, \
    select_kendall_correlated_features


class Preprocessor:
    def __init__(self, data_file_path, target_column):
        self.data_file_path = data_file_path
        self.target_column = target_column

    def preprocess_data(self):
        # Read data
        data = read_data(self.data_file_path)

        # Drop columns with high cardinality
        high_cardinality_columns = get_high_cardinality_columns(
            data, threshold=20)
        data = drop_columns(data, high_cardinality_columns)

        # Find columns with missing values
        columns_with_missing_values = find_columns_with_missing_values(
            data, threshold=0.7)
        data = drop_columns(data, columns_with_missing_values)

        # Drop columns with constant variance
        constant_variance_columns = get_constant_variance_columns(data)
        data = drop_columns(data, constant_variance_columns)

        # Ordinal encode categorical columns
        data = ordinal_encode_categorical_columns(data)

        # Select numerical and categorical columns
        categorical_columns, numerical_columns = find_cat_and_num_columns(
            data
        )

        # Select K best features using chi-square test
        k_best_categorical_features = select_k_best_features(
            data[categorical_columns + [self.target_column]],
            target=self.target_column, k=10)
        k_best_numerical_features = select_kendall_correlated_features(
            data[numerical_columns + [self.target_column]],
            target=self.target_column, k=10)

        # Combine selected features
        selected_features = k_best_categorical_features
        + k_best_numerical_features

        # Final clean data
        clean_data = data[selected_features + [self.target_column]]

        return clean_data