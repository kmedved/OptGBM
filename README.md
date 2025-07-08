# OptGBM

[![Python package](https://github.com/Y-oHr-N/OptGBM/workflows/Python%20package/badge.svg?branch=master)](https://github.com/Y-oHr-N/OptGBM/actions?query=workflow%3A%22Python+package%22)
[![codecov](https://codecov.io/gh/Y-oHr-N/OptGBM/branch/master/graph/badge.svg)](https://codecov.io/gh/Y-oHr-N/OptGBM)
[![PyPI](https://img.shields.io/pypi/v/OptGBM)](https://pypi.org/project/OptGBM/)
[![PyPI - License](https://img.shields.io/pypi/l/OptGBM)](https://pypi.org/project/OptGBM/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Y-oHr-N/OptGBM/master)

OptGBM (= [Optuna](https://optuna.org/) + [LightGBM](http://github.com/microsoft/LightGBM)) provides a scikit-learn compatible estimator that tunes hyperparameters in LightGBM with Optuna.

This package requires Python 3.8 or newer.

## Examples

```python
import optgbm as lgb
from sklearn.datasets import load_diabetes

reg = lgb.LGBMRegressor(random_state=0)
X, y = load_diabetes(return_X_y=True)

reg.fit(X, y)

y_pred = reg.predict(X)
```

By default, the following hyperparameters will be searched.

- `bagging_fraction`
- `bagging_freq`
- `feature_fraction`
- `lambda_l1`
- `lambda_l2`
- `max_depth`
- `min_data_in_leaf`
- `num_leaves`

## Customizing the Search Space

You can take full control of the hyperparameter search by passing a `param_distributions` dictionary to the estimator. The values should be Optuna distribution objects.

```python
import optgbm as lgb
from optuna.distributions import IntDistribution, FloatDistribution
from sklearn.datasets import load_breast_cancer

# Define a custom search space
param_distributions = {
    "num_leaves": IntDistribution(20, 100),
    "learning_rate": FloatDistribution(0.01, 0.2, log=True),
    "lambda_l1": FloatDistribution(1e-8, 1.0, log=True),
}

clf = lgb.LGBMClassifier(
    param_distributions=param_distributions,
    n_trials=50,  # Search more trials for the custom space
    random_state=42,
)

X, y = load_breast_cancer(return_X_y=True)
clf.fit(X, y)
```

## Installation

```
pip install optgbm
```

## Testing

```
tox
```
