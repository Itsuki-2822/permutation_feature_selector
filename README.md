# Permutation Feature Selector
## Acknowledgements
This library is inspired by the functionality and design of **Scikit-learn's permutation importance**.

## Links
PyPI：https://pypi.org/project/PermutationFeatureSelector/

## Installation
To install the Library,

### Standard Installation

You can install the package directly from PyPI using pip. This is the easiest and most common method:
```bash
$ pip install PermutationFeatureSelector
```
### Installation from Source
If you prefer to install from the source or if you want the latest version that may not yet be released on PyPI, you can use the following commands:
```bash
$ git clone https://github.com/Itsuki-2822/PermutationFeatureSelector.git
$ cd PermutationFeatureSelector
$ python setup.py install
```
### For developer
If you are a contributor or if you want to install the latest development version of the library, use the following command to install directly from the GitHub repository:

```bash
$ pip install --upgrade git+https://github.com/Itsuki-2822/PermutationFeatureSelector
```

## What Permutation Importance

### Basic Concept
The calculation of permutation importance proceeds through the following steps:

- #### Evaluate Model Performance:
  - Measure the performance metric (e.g., accuracy or error) of the model using the original dataset before any permutation.
- #### Shuffle the Feature:
  - Randomly shuffle the order of values in one feature of the dataset. This disrupts the relationship between that feature and the target variable.
- #### Re-evaluate Performance:
  - Assess the model's performance again, using the dataset with the shuffled feature.
- #### Calculate Importance:
  - Compute the difference in performance before and after the permutation. A larger difference indicates that the feature is more "important."

### Model-Independent Advantage
Permutation importance is independent of the internal mechanisms of any specific model, which means it does not rely on the evaluation mechanisms specific to models like gradient boosting or decision trees. It can be applied across various predictive models (linear models, decision trees, neural networks, etc.)

### Considerations for Use
- #### Randomness:
  - Since the feature shuffling is a random process, the results may vary slightly each time. To obtain stable evaluations, averaging several assessments is recommended.
- #### Correlated Features:
  - If multiple features are strongly correlated, their importance may be underestimated. Addressing this issue may require careful feature selection and engineering.

## Examples
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import lightgbm as lgb
import matplotlib.pyplot as plt
from PermutationFeatureSelector import PermutationFeatureSelector

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

selector = PermutationFeatureSelector(model, X_test, y_test, metric='rmse', n_repeats=30, random_state=42)

# Calculation of permutation importance
perm_importance = selector.calculate_permutation_importance()
print(perm_importance)

# Permutation Importance Plot
selector.plot_permutation_importance()

# Feature selection (e.g., select features with importance at least 1x the average)
chosen_features, chosen_features_df = selector.choose_feat(threshold_method='mean', threshold_value=1.0)
print(chosen_features)
print(chosen_features_df)

```

## References
#### scikit-learn.org：
- https://scikit-learn.org/stable/modules/permutation_importance.html
- https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
- https://github.com/scikit-learn/scikit-learn/blob/2621573e6/sklearn/inspection/_permutation_importance.py#L111

#### medium.com：
- https://medium.com/@TheDataScience-ProF/permutation-importance-c784e3f8a439

#### hacarus.github.io：
- https://hacarus.github.io/interpretable-ml-book-ja/feature-importance.html
