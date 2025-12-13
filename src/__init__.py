"""
Breiman CART Implementation (1984)
====================================

A pure NumPy/Pandas implementation of Classification and Regression Trees
following the original methodology from Breiman, Friedman, Olshen, and Stone (1984).

Features
--------
- Binary decision trees for classification and regression
- Support for both numerical and categorical features
- Cost-complexity pruning for model selection
- Feature importance calculation
- Model persistence (save/load)
- Comprehensive validation and error handling

Quick Start
-----------
>>> from src import CARTClassifier
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Create sample data
>>> X = pd.DataFrame({'x1': [0, 1, 2, 3], 'x2': [0, 1, 1, 0]})
>>> y = np.array([0, 1, 1, 0])
>>>
>>> # Fit classifier
>>> clf = CARTClassifier(max_depth=3)
>>> clf.fit(X, y)
>>>
>>> # Make predictions
>>> predictions = clf.predict(X)
>>>
>>> # Get feature importance
>>> importances = clf.get_feature_importance()

Classes
-------
CARTClassifier : Classification tree
CARTRegressor : Regression tree
Node : Tree node structure
Splitter : Split finding algorithm
CostComplexityPruner : Pruning algorithm

References
----------
Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
Classification and Regression Trees. Wadsworth.
"""

from .tree import CARTClassifier, CARTRegressor
from .node import Node
from .splitter import Splitter
from .pruning import CostComplexityPruner

__version__ = "0.1.1"
__author__ = "Adam Khald"
__doc_url__ = "https://github.com/Adamkhald/breiman-cart"
__all__ = [
    "CARTClassifier",
    "CARTRegressor",
    "Node",
    "Splitter",
    "CostComplexityPruner"
]

# Module-level documentation
__doc_url__ = "https://github.com/your-repo/breiman-cart"
__license__ = "MIT"

# Version history
__changelog__ = """
Version 2.0.0 (Current)
-----------------------
- Added comprehensive input validation
- Improved performance for categorical features
- Added feature importance calculation
- Added model save/load functionality
- Added extensive logging support
- Improved pruning algorithm efficiency
- Added comprehensive test suite
- Better error messages and handling
- Added tree visualization helpers
- Improved documentation

Version 1.0.0
-------------
- Initial implementation
- Basic classification and regression
- Cost-complexity pruning
- Categorical feature support
"""