# Machine Learning Fundamentals - Lab Projects

Implementation of fundamental machine learning algorithms from scratch, following lab projects from Hui Jiang's *Machine Learning Fundamentals*.

## Overview

This repository contains from-scratch implementations of core ML algorithms using only linear algebra libraries. The goal is to build deep understanding of underlying mechanisms rather than relying on high-level abstractions.

**Implementations available in:** Python (NumPy) and Mathematica

## Completed Projects

### Lab Project I: Feature Extraction
**Dataset:** MNIST handwritten digits

**Implemented Methods:**
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Comparison with t-SNE for visualization

**Key Results:**
- Eigenvalue analysis and variance retention
- 2D/3D projections for data visualization
- Maximum LDA dimensions for multi-class problems

### Lab Project II: Discriminative Classification Models
**Dataset:** MNIST handwritten digits

**Implemented Methods:**
- Linear Regression for binary classification
- Minimum Classification Error (MCE)
- Logistic Regression
- Support Vector Machines (linear and RBF kernel)
  - Primal formulation with projected gradient descent
  - Multi-class classification via one-vs-one strategy

**Optimization Techniques:**
- Full gradient descent
- Mini-batch stochastic gradient descent (SGD)
- Projected gradient descent for constrained optimization

**Key Implementation Details:**
- Custom optimizers (no off-the-shelf solvers)
- Loss normalization strategies
- Regularization for numerical stability
- Bias term handling in predictions

### Lab Project III: Word Representations
**Dataset:** enwik8 text corpus + WordSim353 evaluation

**Implemented Methods:**
- Co-occurrence matrix construction with context windows
- Truncated Singular Value Decomposition (SVD) for dimensionality reduction
- Word2Vec-style embedding learning via alternating optimization
- Stochastic Gradient Descent (SGD) for embedding optimization

**Key Features:**
- Multiple embedding dimensions (20, 50, 100)
- Word similarity evaluation using Spearman correlation
- Memory-efficient sparse matrix operations
- Preprocessing pipeline for large text corpora

### Lab Project IV: Neural Networks
**Dataset:** MNIST handwritten digits

**Implemented Architectures:**
- **Fully Connected Neural Network:**
  - Configurable hidden layer dimensions
  - ReLU activation functions
  - He weight initialization
  - Cross-entropy loss with softmax output

- **Convolutional Neural Network (CNN):**
  - 3 convolutional layers (customizable kernel counts: 32, 64, 64)
  - Fixed 3×3 convolution kernels and 2×2 max pooling
  - Sliding window implementation for convolution operations
  - Custom max pooling with derivative computation
  - 2 fully connected layers (customizable dimensions)

**Implementation Details:**
- Manual backpropagation for both architectures
- Sliding window convolution without convolution libraries
- Learning rate scheduling (decay at epochs 10 and 25)
- Batch processing for memory efficiency
- Numerical stabilization in softmax computations

## Project Structure
```
├── lab_project_1/          # Feature extraction
│   ├── pca.py              # Python implementation
│   ├── pca.nb              # Mathematica notebook
│   └── pca.wls             # Mathematica script
├── lab_project_2/          # Discriminative models
│   ├── linear_regression.py
│   ├── linear_regression.nb
│   ├── logistic_regression.py
│   ├── logistic_regression.nb
│   ├── mce.py
│   ├── mce.nb
│   ├── svm.py
│   └── svm.nb
├── lab_project_3/          # Word representations
│   ├── word_representations.py
│   └── word_representations.nb
└── lab_project_4/          # Neural networks
    ├── fully_connected_nn.py
    ├── fully_connected_nn.nb
    ├── cnn.py
    └── cnn.nb
```

## Installation

### Python Dependencies
```bash
# Core dependencies for all projects
pip install numpy matplotlib

# Additional dependencies for specific projects
pip install unidecode scipy scikit-learn  # Lab Project 3: Word representations
```

### Data Download
**MNIST data** is automatically downloaded on first run from:
- https://github.com/fgnt/mnist/

**For Lab Project 3 (Word Representations):**
- `enwik8` text corpus (download manually from http://mattmahoney.net/dc/enwik8.zip)
- `wordsim353crowd.csv` evaluation dataset (included or download from WordSim-353)

## Usage

### Python Implementations

Each implementation can be run directly and includes configuration at the top:

```python
# Example: Running SVM classifier
CONFIG = {
    'digits': [5, 8],
    'learning_rate': 1e-4,
    'max_iterations': 100000,
    'convergence_tolerance': 1e-6,
    'batch_size': 64,
}
```

```python
# Example: Running CNN
CONFIG = {
    'epochs': 50,
    'learning_rate': 1e-2,
    'batch_size': 77,
}
```

```bash
# Run individual implementations
python lab_project_2/svm.py
python lab_project_3/word_representations.py
python lab_project_4/cnn.py
python lab_project_4/fully_connected_nn.py
```

### Mathematica Implementations

Open `.nb` notebooks in Mathematica, or run scripts:
```bash
wolframscript -file lab_project_1/pca.wls
```

## Design Principles

1. **First-principles implementation:** Build from mathematical foundations using only linear algebra operations
2. **No ML libraries:** Implementations use only NumPy/Mathematica for matrix operations

## Technical Notes

### Known Issues
- NumPy 2.x warnings on ARM64/Apple Silicon (benign, results unaffected)
- Some dependencies in older RL codebases may need manual resolution

## In Progress

- Lab Project V: Ensemble Learning
- Lab Project VI: Multivariate Gaussian Models

## References

Jiang, H. (2021). *Machine Learning Fundamentals: A Concise Introduction*. Cambridge University Press.

## License

This is educational code for learning purposes.

---

**Note:** For production use, consider established libraries like scikit-learn or PyTorch.
