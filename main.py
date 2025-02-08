import numpy as np
import pandas as pd
from tree.random_forest import RandomForestClassifier
from tree.utils import train_test_split, accuracy_score, classification_report
from tree.model_io import save_model_and_metadata, load_model_and_prepare_data
from time import time
import logging
import argparse
import os

# Set logging level to INFO to see detailed progress
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or load a random forest model for DDoS detection"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load"],
        default="train",
        help="Whether to train a new model or load an existing one",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/random_forest.pkl",
        help="Path to save/load the model",
    )

    # Random Forest specific parameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of each tree",
    )

    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split a node",
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required in a leaf node",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Number of features to consider for best split",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs. If None, uses all available cores",
    )

    parser.add_argument(
        "--tree-type",
        type=str,
        choices=["cart", "c4.5"],
        default="cart",
        help="Type of decision tree to use in the forest",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion of data to use for testing",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )

    return parser.parse_args()


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    # read first 30000 rows of the dataset
    raw_data = pd.read_csv(
        "data/Wednesday-workingHours.pcap_ISCX.csv",
        low_memory=False,
        nrows=20000,
        skiprows=range(1, 20000),
    )

    data = raw_data
    # filter out the rows with nan values
    data = data.dropna()
    y = data["Label"]
    x = data.drop(columns=["Label", "Flow Bytes/s"])

    # keep first 10 features of x
    x = x.iloc[:, :10]

    return x, y


def train_model(args, x, y):
    """Train a new random forest model with the given parameters."""
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state
    )

    # Initialize Random Forest
    forest = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        tree_type=args.tree_type,
        random_state=args.random_state,
    )

    # Train the model
    print("\nTraining Random Forest...")
    start_time = time()
    forest.fit(x_train, y_train)
    train_time = time() - start_time

    # Make predictions on training set
    y_train_pred = forest.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Save model with metadata
    model_params = forest.get_params()

    training_metrics = {
        "training_time": f"{train_time:.2f} seconds",
        "training_accuracy": f"{train_accuracy:.4f}",
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "n_trees": args.n_estimators,
        "n_jobs": args.n_jobs if args.n_jobs is not None else "all cores",
    }

    save_model_and_metadata(
        model=forest,
        model_path=args.model_path,
        feature_names=list(x.columns),
        model_params=model_params,
        training_metrics=training_metrics,
    )

    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Training Accuracy: {train_accuracy:.4f}")

    return forest, x_test, y_test


def evaluate_model(model, x_test, y_test):
    """Evaluate the model's performance."""
    # Make predictions
    print("\nMaking predictions...")
    start_time = time()
    y_pred = model.predict(x_test)
    predict_time = time() - start_time
    print(f"Prediction completed in {predict_time:.2f} seconds")

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Print detailed classification report
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load and preprocess data
    x, y = load_and_preprocess_data()

    if args.mode == "train":
        # Train new model
        model, x_test, y_test = train_model(args, x, y)
    else:
        # Load existing model and prepare test data
        try:
            model, x_test, y_test = load_model_and_prepare_data(
                model_path=args.model_path,
                x=x,
                y=y,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        except FileNotFoundError:
            print(
                f"No model found at {args.model_path}. Please train a new model first."
            )
            exit(1)

    # Evaluate model
    evaluate_model(model, x_test, y_test)
