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

## Project Structure
```
├── lab_project_1/          # Feature extraction
│   ├── pca.py              # Python implementation
│   ├── pca.nb              # Mathematica notebook
│   └── pca.wls             # Mathematica script
└── lab_project_2/          # Discriminative models
    ├── linear_regression.py
    ├── linear_regression.nb
    ├── logistic_regression.py
    ├── logistic_regression.nb
    ├── mce.py
    ├── mce.nb
    ├── svm.py
    └── svm.nb
```

## Installation

### Python Dependencies
```bash
pip install numpy matplotlib
```

### Data Download
MNIST data is automatically downloaded on first run from:
- https://github.com/fgnt/mnist/

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
```bash
python lab_project_2/svm.py
```

### Mathematica Implementations

Open `.nb` notebooks in Mathematica, or run scripts:
```bash
wolframscript -file lab_project_1/pca.wls
```

## Design Principles

1. **First-principles implementation:** Build from mathematical foundations using only linear algebra operations
2. **No ML libraries:** Implementations use only NumPy/Mathematica for matrix operations
3. **Systematic debugging:** Adversarial testing and methodical issue isolation
4. **Learning velocity:** Balance between depth and forward progress

## Technical Notes

### Known Issues
- NumPy 2.x warnings on ARM64/Apple Silicon (benign, results unaffected)
- Some dependencies in older RL codebases may need manual resolution

### Performance
- Mathematica parallel implementations available for computationally intensive operations
- Python implementations prioritize clarity over optimization

## In Progress

- Lab Project III: Natural Language Processing
- Lab Project IV: Neural Networks
- Lab Project V: Ensemble Learning
- Lab Project VI: Multivariate Gaussian Models

## References

Jiang, H. (2021). *Machine Learning Fundamentals: A Concise Introduction*. Cambridge University Press.

## License

This is educational code for learning purposes.

---

**Note:** These implementations prioritize educational value and understanding over production-ready performance. For production use, consider established libraries like scikit-learn or PyTorch.
