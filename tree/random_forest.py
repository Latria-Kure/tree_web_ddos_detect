import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Type, Literal
from multiprocessing import Pool, cpu_count
import logging
from collections import Counter
import pickle
import os

from .decision_tree import CARTDecisionTree, C45DecisionTree, BaseDecisionTree
from .utils import train_test_split

logger = logging.getLogger(__name__)


def _train_single_tree(args: tuple) -> BaseDecisionTree:
    """Helper function to train a single tree for parallel processing.

    Parameters
    ----------
    args : tuple
        Tuple containing (X, y, tree_params, tree_class, bootstrap_indices)

    Returns
    -------
    BaseDecisionTree
        Trained decision tree
    """
    X, y, tree_params, tree_class, indices = args
    # Create and train tree with bootstrapped data
    tree = tree_class(**tree_params)
    tree.fit(X.iloc[indices], y.iloc[indices])
    return tree


class RandomForestClassifier:
    """Random Forest Classifier implementation using parallel processing."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        bootstrap: bool = True,
        n_jobs: Optional[int] = None,
        tree_type: Literal["cart", "c4.5"] = "cart",
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of each tree
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required in a leaf node
        max_features : int, optional
            Number of features to consider for best split. If None, uses sqrt(n_features)
        bootstrap : bool, default=True
            Whether to use bootstrap samples
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available cores
        tree_type : str, default="cart"
            Type of decision tree to use ("cart" or "c4.5")
        random_state : int, optional
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()
        self.tree_type = tree_type.lower()
        self.random_state = random_state
        self.trees: List[BaseDecisionTree] = []

        # Select tree class based on type
        self.tree_class = (
            CARTDecisionTree if self.tree_type == "cart" else C45DecisionTree
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the random forest using parallel processing.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target values
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(X)

        # Set max_features if not specified
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))

        # Prepare parameters for each tree
        tree_params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "validation_split": 0.0,  # No validation split for individual trees
            "random_state": None,  # Will be set differently for each tree
        }

        # Generate bootstrap indices for all trees
        bootstrap_indices = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            bootstrap_indices.append(indices)

        # Prepare arguments for parallel processing
        train_args = [
            (X, y, tree_params, self.tree_class, indices)
            for indices in bootstrap_indices
        ]

        # Train trees in parallel
        self.logger.info(
            f"Training {self.n_estimators} {self.tree_type.upper()} trees using {self.n_jobs} processes..."
        )
        with Pool(processes=self.n_jobs) as pool:
            self.trees = pool.map(_train_single_tree, train_args)

        self.logger.info("Random forest training completed")

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predict class labels using majority voting.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict

        Returns
        -------
        List[str]
            Predicted class labels
        """
        # Get predictions from all trees
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))

        # Convert to numpy array for easier manipulation
        predictions = np.array(predictions)

        # Use majority voting for final predictions
        final_predictions = []
        for i in range(len(X)):
            tree_predictions = predictions[:, i]
            # Get most common prediction
            most_common = Counter(tree_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)

        return final_predictions

    def save_model(self, filepath: str) -> None:
        """Save the random forest model.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.logger.info(f"Saving random forest model to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("Model saved successfully")

    @classmethod
    def load_model(cls, filepath: str) -> "RandomForestClassifier":
        """Load a saved random forest model.

        Parameters
        ----------
        filepath : str
            Path to the saved model

        Returns
        -------
        RandomForestClassifier
            The loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise ValueError(f"Loaded model is not an instance of {cls.__name__}")

        model.logger = logging.getLogger(model.__class__.__name__)
        model.logger.info(f"Model loaded successfully from {filepath}")
        return model

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "n_jobs": self.n_jobs,
            "tree_type": self.tree_type,
            "random_state": self.random_state,
        }
