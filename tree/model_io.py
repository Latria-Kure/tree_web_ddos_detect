import os
import logging
from typing import Optional, Tuple
import pandas as pd
from .random_forest import RandomForestClassifier
from .utils import train_test_split

logger = logging.getLogger(__name__)


def save_model_and_metadata(
    model: RandomForestClassifier,
    model_path: str,
    feature_names: list,
    model_params: dict,
    training_metrics: Optional[dict] = None,
) -> None:
    """Save model along with metadata about features and parameters.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained model to save
    model_path : str
        Path where to save the model
    feature_names : list
        Names of features used in training
    model_params : dict
        Parameters used to train the model
    training_metrics : dict, optional
        Training metrics like accuracy, training time etc.
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    model.save_model(model_path)

    # Save metadata
    metadata_path = model_path.rsplit(".", 1)[0] + "_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("Model Metadata:\n")
        f.write("--------------\n\n")

        f.write("Features Used:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"{i}. {feature}\n")
        f.write("\n")

        f.write("Model Parameters:\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")

        if training_metrics:
            f.write("Training Metrics:\n")
            for metric, value in training_metrics.items():
                f.write(f"{metric}: {value}\n")

    logger.info(f"Model and metadata saved to {os.path.dirname(model_path)}")


def load_model_and_prepare_data(
    model_path: str,
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """Load a saved model and prepare test data.

    Parameters
    ----------
    model_path : str
        Path to the saved model
    x : pd.DataFrame
        Feature data
    y : pd.Series
        Target labels
    test_size : float, default=0.25
        Proportion of data to use for testing
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    model : RandomForestClassifier
        The loaded model
    x_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    # Load the model
    model = RandomForestClassifier.load_model(model_path)

    # Split data for testing
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Check if metadata exists and log it
    metadata_path = model_path.rsplit(".", 1)[0] + "_metadata.txt"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            logger.info("Model metadata:\n" + f.read())

    return model, x_test, y_test
