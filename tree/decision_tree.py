import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging
import pickle
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Node:
    """Base node class for decision trees"""

    def __init__(
        self,
        feature: Optional[str] = None,
        threshold: Optional[float] = None,
        value: Optional[str] = None,
        children: Optional[Dict[str, "Node"]] = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.children = children or {}

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class BaseDecisionTree(ABC):
    """Abstract base class for decision tree classifiers"""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        validation_split: float = 0.2,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.validation_split = validation_split
        self.random_state = random_state
        self.root: Optional[Node] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file.

        Parameters
        ----------
        filepath : str
            Path where the model should be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.logger.info(f"Saving model to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("Model saved successfully")

    @classmethod
    def load_model(cls, filepath: str) -> "BaseDecisionTree":
        """Load a trained model from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file

        Returns
        -------
        BaseDecisionTree
            The loaded decision tree model
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

    @abstractmethod
    def _calculate_split_metric(self, y: pd.Series) -> float:
        """Calculate impurity metric (entropy for C4.5, gini for CART)"""
        pass

    @abstractmethod
    def _find_best_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Optional[str], Optional[float], float]:
        """Find the best feature and split point"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the decision tree to the training data"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.logger.info(
            f"Starting tree construction with {len(X)} samples and {len(X.columns)} features"
        )
        self.logger.info(f"Initial class distribution: {dict(y.value_counts())}")
        self.logger.info(
            f"Parameters: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, min_samples_leaf={self.min_samples_leaf}"
        )

        # Split data for validation if needed
        if self.validation_split > 0:
            train_size = int(len(X) * (1 - self.validation_split))
            indices = np.random.permutation(len(X))
            train_idx, val_idx = indices[:train_size], indices[train_size:]

            self.X_train = X.iloc[train_idx]
            self.y_train = y.iloc[train_idx]
            self.X_val = X.iloc[val_idx]
            self.y_val = y.iloc[val_idx]

            self.logger.info(
                f"Split data into {len(self.X_train)} training samples and {len(self.X_val)} validation samples"
            )
        else:
            self.X_train = X
            self.y_train = y
            self.X_val = None
            self.y_val = None

        self.logger.info("Starting tree construction...")
        self.root = self._build_tree(self.X_train, self.y_train)
        self.logger.info("Tree construction completed")

        if self.validation_split > 0:
            self.logger.info("Starting post-pruning...")
            self._post_prune(self.root)
            self.logger.info("Post-pruning completed")

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predict class labels for samples in X"""
        if self.root is None:
            raise ValueError("Tree has not been fitted yet")
        return [self._traverse_tree(x, self.root) for _, x in X.iterrows()]

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        """Recursively build the decision tree"""
        n_samples = len(y)
        n_classes = len(y.unique())

        self.logger.info(f"Building node at depth {depth} with {n_samples} samples")
        self.logger.info(f"Class distribution: {dict(y.value_counts())}")

        # Check stopping criteria
        if self.max_depth is not None and depth >= self.max_depth:
            self.logger.info(f"Stopping: reached maximum depth {self.max_depth}")
            return Node(value=y.mode()[0])
        if n_samples < self.min_samples_split:
            self.logger.info(
                f"Stopping: too few samples to split ({n_samples} < {self.min_samples_split})"
            )
            return Node(value=y.mode()[0])
        if n_classes == 1:
            self.logger.info("Stopping: pure node reached")
            return Node(value=y.mode()[0])

        # Find best split
        feature, threshold, best_metric = self._find_best_split(X, y)
        if feature is None:
            self.logger.info("Stopping: no valid split found")
            return Node(value=y.mode()[0])

        self.logger.info(
            f"Selected feature: {feature}"
            + (f" with threshold: {threshold:.4f}" if threshold is not None else "")
        )

        # Create child nodes
        node = Node(feature=feature, threshold=threshold)

        # Split the data
        if pd.api.types.is_numeric_dtype(X[feature]):
            left_mask = X[feature] <= threshold
            splits = {
                "left": (X[left_mask], y[left_mask]),
                "right": (X[~left_mask], y[~left_mask]),
            }
            self.logger.info(
                f"Split sizes - Left: {sum(left_mask)}, Right: {sum(~left_mask)}"
            )
        else:
            splits = {
                str(value): (X[X[feature] == value], y[X[feature] == value])
                for value in X[feature].unique()
            }
            split_sizes = {k: len(v[1]) for k, v in splits.items()}
            self.logger.info(f"Split sizes by category: {split_sizes}")

        # Build child nodes
        for split_value, (split_X, split_y) in splits.items():
            if len(split_y) >= self.min_samples_leaf:
                self.logger.info(f"Building {split_value} child node...")
                node.children[split_value] = self._build_tree(
                    split_X, split_y, depth + 1
                )
            else:
                self.logger.info(
                    f"Creating leaf node for {split_value} (too few samples: {len(split_y)} < {self.min_samples_leaf})"
                )
                node.children[split_value] = Node(value=y.mode()[0])

        return node

    def _traverse_tree(self, x: pd.Series, node: Node) -> str:
        """Traverse the tree to make a prediction"""
        if node.is_leaf:
            return node.value

        feature_value = x[node.feature]
        if pd.api.types.is_numeric_dtype(type(feature_value)):
            child_key = "left" if feature_value <= node.threshold else "right"
        else:
            child_key = str(feature_value)

        if child_key not in node.children:
            # Handle unseen categories by returning most common class
            return max(
                (child.value for child in node.children.values() if child.is_leaf),
                key=lambda x: sum(
                    1
                    for child in node.children.values()
                    if child.is_leaf and child.value == x
                ),
            )

        return self._traverse_tree(x, node.children[child_key])

    def _post_prune(self, node: Node) -> bool:
        """Recursively prune the tree using reduced error pruning"""
        if node.is_leaf:
            return False

        # Recursively prune children
        any_pruned = False
        for child in node.children.values():
            if self._post_prune(child):
                any_pruned = True

        if any_pruned:
            return False

        # Calculate accuracy before pruning
        pre_prune_acc = self._get_node_accuracy(node)

        # Try pruning
        original_state = (node.feature, node.threshold, node.children)
        node.feature = None
        node.threshold = None
        node.children = {}
        node.value = self._get_majority_class(node)

        # Calculate accuracy after pruning
        post_prune_acc = self._get_node_accuracy(node)

        # Revert if accuracy decreased
        if post_prune_acc <= pre_prune_acc:
            node.feature, node.threshold, node.children = original_state
            node.value = None
            return False

        self.logger.info(
            f"Pruned node {node.feature} - Accuracy improved: {pre_prune_acc:.4f} -> {post_prune_acc:.4f}"
        )
        return True

    def _get_node_accuracy(self, node: Node) -> float:
        """Calculate accuracy for samples reaching this node"""
        if self.X_val is None or self.y_val is None:
            return 0.0

        mask = self._get_samples_reaching_node(node)
        if not any(mask):
            return 0.0

        predictions = [
            self._traverse_tree(self.X_val.iloc[i], node)
            for i in range(len(self.X_val))
            if mask[i]
        ]
        actual = self.y_val[mask]
        return np.mean(np.array(predictions) == actual.values)

    def _get_samples_reaching_node(self, target_node: Node) -> np.ndarray:
        """Get boolean mask of validation samples reaching the node"""

        def _traverse_to_node(x: pd.Series, node: Node) -> bool:
            if node is target_node:
                return True
            if node.is_leaf:
                return False

            feature_value = x[node.feature]
            if pd.api.types.is_numeric_dtype(type(feature_value)):
                child_key = "left" if feature_value <= node.threshold else "right"
            else:
                child_key = str(feature_value)

            if child_key not in node.children:
                return False

            return _traverse_to_node(x, node.children[child_key])

        return np.array(
            [_traverse_to_node(row, self.root) for _, row in self.X_val.iterrows()]
        )

    def _get_majority_class(self, node: Node) -> str:
        """Get majority class of validation samples reaching the node"""
        if self.X_val is None or self.y_val is None:
            return self.y_train.mode()[0]

        mask = self._get_samples_reaching_node(node)
        if not any(mask):
            return self.y_val.mode()[0]

        return self.y_val[mask].mode()[0]


class C45DecisionTree(BaseDecisionTree):
    """C4.5 decision tree classifier"""

    def _calculate_split_metric(self, y: pd.Series) -> float:
        """Calculate entropy for C4.5"""
        counts = y.value_counts()
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _find_best_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Optional[str], Optional[float], float]:
        """Find best split using information gain ratio"""
        best_gain_ratio = 0
        best_feature = None
        best_threshold = None

        self.logger.debug(f"Finding best split for {len(y)} samples")

        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]):
                gain_ratio, threshold = self._find_numeric_split(X[feature], y)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold
            else:
                gain_ratio = self._find_categorical_split(X[feature], y)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = None

        return best_feature, best_threshold, best_gain_ratio

    def _find_numeric_split(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Find best split point for numeric feature using gain ratio"""
        best_gain_ratio = 0
        best_threshold = None

        # Get unique sorted values
        unique_values = np.sort(x.unique())
        # Calculate potential thresholds as midpoints
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        parent_entropy = self._calculate_split_metric(y)

        for threshold in thresholds:
            left_mask = x <= threshold
            left_y = y[left_mask]
            right_y = y[~left_mask]

            # Skip if split creates empty nodes
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Calculate information gain
            left_entropy = self._calculate_split_metric(left_y)
            right_entropy = self._calculate_split_metric(right_y)

            weighted_entropy = (
                len(left_y) / len(y) * left_entropy
                + len(right_y) / len(y) * right_entropy
            )
            info_gain = parent_entropy - weighted_entropy

            # Calculate split information
            split_info = -sum(
                count / len(y) * np.log2(count / len(y))
                for count in [len(left_y), len(right_y)]
            )

            # Calculate gain ratio
            gain_ratio = info_gain / split_info if split_info != 0 else 0

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold

        return best_gain_ratio, best_threshold

    def _find_categorical_split(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate gain ratio for categorical feature"""
        parent_entropy = self._calculate_split_metric(y)
        weighted_entropy = 0
        split_info = 0

        for value in x.unique():
            subset_y = y[x == value]
            prob = len(subset_y) / len(y)
            weighted_entropy += prob * self._calculate_split_metric(subset_y)
            split_info -= prob * np.log2(prob)

        info_gain = parent_entropy - weighted_entropy
        return info_gain / split_info if split_info != 0 else 0


class CARTDecisionTree(BaseDecisionTree):
    """CART decision tree classifier"""

    def _calculate_split_metric(self, y: pd.Series) -> float:
        """Calculate Gini impurity"""
        counts = y.value_counts()
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _find_best_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Optional[str], Optional[float], float]:
        """Find best split using Gini index"""
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        self.logger.debug(f"Finding best split for {len(y)} samples")

        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]):
                gini, threshold = self._find_numeric_split(X[feature], y)
            else:
                gini, threshold = self._find_categorical_split(X[feature], y)

            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _find_numeric_split(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Find best split point for numeric feature using Gini index"""
        best_gini = float("inf")
        best_threshold = None

        unique_values = np.sort(x.unique())
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in thresholds:
            left_mask = x <= threshold
            left_y = y[left_mask]
            right_y = y[~left_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gini = len(left_y) / len(y) * self._calculate_split_metric(left_y) + len(
                right_y
            ) / len(y) * self._calculate_split_metric(right_y)

            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold

        return best_gini, best_threshold

    def _find_categorical_split(self, x: pd.Series, y: pd.Series) -> Tuple[float, None]:
        """Calculate Gini index for categorical feature"""
        weighted_gini = 0
        for value in x.unique():
            subset_y = y[x == value]
            prob = len(subset_y) / len(y)
            weighted_gini += prob * self._calculate_split_metric(subset_y)
        return weighted_gini, None
