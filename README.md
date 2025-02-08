# Web DDoS Detection with Random Forest

A Python implementation of a Random Forest classifier for Web DDoS (Distributed Denial of Service) attack detection. This implementation includes both CART and C4.5 decision tree algorithms with parallel processing capabilities.

## Features

- Custom implementation of Random Forest classifier
- Support for both CART and C4.5 decision trees
- Parallel processing for faster training
- Model persistence (save/load functionality)
- Comprehensive evaluation metrics
- Detailed logging of training progress
- Command-line interface for easy usage

## Requirements

- Python 3.8+
- NumPy
- Pandas

## Dataset

The ISCX IDS 2017 dataset used in this project can be downloaded from:
[http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/)

After downloading, place the CSV files in the `data/` directory. The main script uses the `Wednesday-workingHours.pcap_ISCX.csv` file by default.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tree_web_ddos_detect.git
cd tree_web_ddos_detect
```

2. Install dependencies:
```bash
pip install numpy pandas
```

## Usage

### Command Line Interface

The main script provides various options for training and evaluating models:

```bash
python main.py --mode train --tree-type cart --n-estimators 100
```

### Command Line Options

- `--mode`: Choose between 'train' or 'load' [default: train]
- `--model-path`: Path to save/load model [default: models/random_forest.pkl]
- `--tree-type`: Type of decision tree to use ('cart' or 'c4.5') [default: cart]
- `--n-estimators`: Number of trees in the forest [default: 100]
- `--max-depth`: Maximum depth of each tree [default: None]
- `--min-samples-split`: Minimum samples required to split [default: 2]
- `--min-samples-leaf`: Minimum samples in leaf nodes [default: 1]
- `--max-features`: Number of features to consider for splits [default: sqrt(n_features)]
- `--n-jobs`: Number of parallel jobs [default: all cores]
- `--test-size`: Proportion of data for testing [default: 0.25]
- `--random-state`: Random seed for reproducibility [default: 42]

### Examples

1. Train a new Random Forest with CART trees:
```bash
python main.py --mode train --tree-type cart --n-estimators 200 --max-depth 15
```

2. Train using C4.5 trees with custom parameters:
```bash
python main.py --mode train --tree-type c4.5 --n-estimators 150 --min-samples-split 5
```

3. Load and evaluate an existing model:
```bash
python main.py --mode load --model-path models/my_model.pkl
```

## Project Structure

```
tree_web_ddos_detect/
├── tree/
│   ├── __init__.py
│   ├── decision_tree.py    # Base, CART, and C4.5 implementations
│   ├── random_forest.py    # Random Forest implementation
│   ├── model_io.py        # Model persistence utilities
│   └── utils.py           # Evaluation metrics and utilities
├── main.py                # Command line interface
├── models/                # Directory for saved models
└── data/                  # Directory for datasets
```

## Implementation Details

### Random Forest
- Supports both CART and C4.5 decision trees
- Uses bootstrap sampling for tree training
- Implements parallel processing for tree construction
- Features majority voting for predictions

### Decision Trees
- CART (Classification and Regression Trees)
  - Uses Gini impurity for splits
  - Supports both numeric and categorical features
  
- C4.5
  - Uses information gain ratio for splits
  - Handles both continuous and discrete attributes
  - Includes tree pruning capabilities

### Model Persistence
- Saves both model and metadata
- Metadata includes:
  - Features used
  - Model parameters
  - Training metrics
  - Performance statistics

## Evaluation Metrics

The implementation provides comprehensive evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Support
- Detailed classification report

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the original CART and C4.5 algorithms
- Inspired by scikit-learn's Random Forest implementation
- Uses the ISCX IDS 2017 dataset for DDoS detection 