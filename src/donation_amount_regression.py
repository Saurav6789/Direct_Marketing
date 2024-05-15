from preprocess import Preprocessor
from model import XGBoostConfig, CustomXGBoostRegressor
import pandas as pd

if __name__ == "__main__":
    # Paths to train and test data files
    train_file_path = "../data/raw/cup98LRN.txt"
    test_file_path = "../data/raw/cup98VAL.txt"
    output_file_path = "../data/processed/test_data_clf.csv"
    output_file_path_reg = "../data/processed/test_data_reg.csv"
    # Target column name
    target_column = "TARGET_D"

    # Preprocess train data
    train_preprocessor = Preprocessor(
        data_file_path=train_file_path, target_column=target_column
    )
    train_data = train_preprocessor.preprocess_data()
    test_df_clf = pd.read_csv(output_file_path)
    filtered_train_df = train_data[train_data["TARGET_B"] == 1]
    filtered_train_df["TARGET_D"] = train_data[train_data["TARGET_B"] == 1][
        target_column
    ]
    filtered_train_df = filtered_train_df.drop(columns=["TARGET_B"])
    X_filtered = filtered_train_df.drop(columns=[target_column])
    object_cols = X_filtered.select_dtypes(include=["object"]).columns
    X_filtered[object_cols] = X_filtered[object_cols].astype("category")
    y_filtered = filtered_train_df[target_column]
    xgb_config = XGBoostConfig()
    xgb_regressor = CustomXGBoostRegressor(config=xgb_config)

    # Prediction on the test data
    test_df_reg = test_df_clf[test_df_clf["IsDonated"] == 1]
    test_df_reg = test_df_reg.drop(columns=["IsDonated"])
    object_cols = test_df_reg.select_dtypes(include=["object"]).columns
    test_df_reg[object_cols] = test_df_reg[object_cols].astype("category")
    predictions = xgb_regressor.predict(test_df_reg)
    test_df_reg["AMOUNT_Donated"] = predictions
    test_df_reg.to_csv(output_file_path_reg, index=False, mode="w")
