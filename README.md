# Permutation Feature Selector
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

## References
#### scikit-learn.org：
- https://scikit-learn.org/stable/modules/permutation_importance.html
- https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

#### medium.com：
- https://medium.com/@TheDataScience-ProF/permutation-importance-c784e3f8a439

#### hacarus.github.io：
- https://hacarus.github.io/interpretable-ml-book-ja/feature-importance.html
