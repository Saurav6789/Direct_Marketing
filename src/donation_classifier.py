from preprocess import Preprocessor
from model import XGBoostConfig, CustomXGBoostClassifier
from metrics import calculate_classification_metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    # Paths to train and test data files
    train_file_path = "../data/raw/cup98LRN.txt"
    test_file_path = "../data/raw/cup98VAL.txt"
    output_file_path = "../data/processed"
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
    test_data_clf = test_data.copy()

    # Separate features and target
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]
    X.columns = X.columns.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize and train the XGBoost classifier
    xgb_config = XGBoostConfig()
    xgb_classifier = CustomXGBoostClassifier(config=xgb_config)
    cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=10,
                                scoring='roc_auc')
    xgb_classifier.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

    # Calculate classification metrics
    metrics = calculate_classification_metrics(y_test,
                                               y_pred_proba)

    # Print metrics
    print("Classification Metrics:")
    print(metrics)
    # Make Predictions
    predictions = xgb_classifier.predict(test_data_clf)
    test_data_clf["IsDonated"] = predictions
    test_data_clf.to_csv(output_file_path, index=False, mode="w")
