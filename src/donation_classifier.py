from preprocess import Preprocessor
from model import XGBoostConfig, CustomXGBoostClassifier
from metrics import calculate_classification_metrics

if __name__ == "__main__":
    # Paths to train and test data files
    train_file_path = "data/raw/cup98LRN.txt"
    test_file_path = "data/raw/cup98VAL.txt"

    # Target column name
    target_column = "TARGET_B"

    # Preprocess train data
    train_preprocessor = Preprocessor(data_file_path=train_file_path,
                                      target_column=target_column)
    train_data = train_preprocessor.preprocess_data()

    # Preprocess test data
    test_preprocessor = Preprocessor(data_file_path=test_file_path,
                                     target_column=None)
    test_data = test_preprocessor.preprocess_data()

    # Separate features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])

    # Initialize and train the XGBoost classifier
    xgb_config = XGBoostConfig()
    xgb_classifier = CustomXGBoostClassifier(config=xgb_config)
    xgb_classifier.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

    # Calculate classification metrics
    metrics = calculate_classification_metrics(test_data[target_column],
                                               y_pred_proba)

    # Print metrics
    print("Classification Metrics:")
    print(metrics)
