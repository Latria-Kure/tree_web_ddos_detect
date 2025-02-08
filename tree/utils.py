import numpy as np
import pandas as pd
from typing import Tuple, TypeVar, List, Dict, Union

T = TypeVar("T", pd.DataFrame, pd.Series, np.ndarray)


def train_test_split(
    X: T, y: T, test_size: float = 0.25, random_state: int | None = None
) -> Tuple[T, T, T, T]:
    """Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    test_size : float, default=0.25
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to
        include in the test split.
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Returns the train-test split of inputs.
    """
    if not 0 <= test_size <= 1:
        raise ValueError(
            f"test_size should be between 0 and 1, got value = {test_size}"
        )

    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Get number of samples
    n_samples = len(X)

    # Calculate number of test samples
    n_test = int(test_size * n_samples)

    # Create random permutation of indices
    indices = np.random.permutation(n_samples)

    # Split indices into training and test sets
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Handle different input types
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy_score(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
) -> float:
    """Calculate accuracy score.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Accuracy score.
    """
    # Convert inputs to lists if they're not already
    if isinstance(y_true, (np.ndarray, pd.Series)):
        y_true = y_true.tolist()
    if isinstance(y_pred, (np.ndarray, pd.Series)):
        y_pred = y_pred.tolist()

    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must be equal")

    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)


def precision_recall_fscore_support(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
    """Calculate precision, recall, F1-score, and support for each class.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    precision_dict : dict
        Precision scores for each class.
    recall_dict : dict
        Recall scores for each class.
    f1_dict : dict
        F1 scores for each class.
    support_dict : dict
        Number of samples for each class.
    """
    # Convert inputs to lists if they're not already
    if isinstance(y_true, (np.ndarray, pd.Series)):
        y_true = y_true.tolist()
    if isinstance(y_pred, (np.ndarray, pd.Series)):
        y_pred = y_pred.tolist()

    # Get unique classes
    classes = sorted(set(y_true))

    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    support_dict = {}

    for cls in classes:
        # Calculate true positives, false positives, false negatives
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        support = sum(1 for true in y_true if true == cls)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precision_dict[cls] = precision
        recall_dict[cls] = recall
        f1_dict[cls] = f1
        support_dict[cls] = support

    return precision_dict, recall_dict, f1_dict, support_dict


def classification_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
) -> str:
    """Generate a classification report with precision, recall, and F1 score.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    str
        Text summary of the precision, recall, F1 score for each class.
    """
    # Get metrics for each class
    precision_dict, recall_dict, f1_dict, support_dict = (
        precision_recall_fscore_support(y_true, y_pred)
    )

    # Initialize the report string
    report = "\nClassification Report:\n"
    report += f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n"
    report += "-" * 63 + "\n"

    # Add metrics for each class
    classes = sorted(precision_dict.keys())
    total_support = 0

    for cls in classes:
        precision = precision_dict[cls]
        recall = recall_dict[cls]
        f1 = f1_dict[cls]
        support = support_dict[cls]
        total_support += support

        report += f"{str(cls):<15} {precision:>11.4f} {recall:>11.4f} {f1:>11.4f} {support:>11d}\n"

    # Calculate and add macro averages
    n_classes = len(classes)
    macro_precision = sum(precision_dict.values()) / n_classes
    macro_recall = sum(recall_dict.values()) / n_classes
    macro_f1 = sum(f1_dict.values()) / n_classes

    report += "\n"
    report += f"{'Macro Avg':<15} {macro_precision:>11.4f} {macro_recall:>11.4f} "
    report += f"{macro_f1:>11.4f} {total_support:>11d}\n"

    # Calculate and add accuracy
    accuracy = accuracy_score(y_true, y_pred)
    report += f"{'Accuracy':<15} {accuracy:>11.4f}\n"

    return report
