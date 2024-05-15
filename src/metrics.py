from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score


def calculate_classification_metrics(y_true, y_pred_proba):
    """
    Calculate classification metrics including accuracy, precision, recall,
    and ROC AUC score.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities of positive class.

    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }
    return metrics
