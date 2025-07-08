# OptGBM

[![Python package](https://github.com/Y-oHr-N/OptGBM/workflows/Python%20package/badge.svg?branch=master)](https://github.com/Y-oHr-N/OptGBM/actions?query=workflow%3A%22Python+package%22)
[![codecov](https://codecov.io/gh/Y-oHr-N/OptGBM/branch/master/graph/badge.svg)](https://codecov.io/gh/Y-oHr-N/OptGBM)
[![PyPI](https://img.shields.io/pypi/v/OptGBM)](https://pypi.org/project/OptGBM/)
[![PyPI - License](https://img.shields.io/pypi/l/OptGBM)](https://pypi.org/project/OptGBM/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Y-oHr-N/OptGBM/master)

OptGBM (= [Optuna](https://optuna.org/) + [LightGBM](http://github.com/microsoft/LightGBM)) provides a scikit-learn compatible estimator that tunes hyperparameters in LightGBM with Optuna.

This package requires Python 3.8 or newer.

## Usage Overview

The package exposes `LGBMClassifier` and `LGBMRegressor` classes that behave
like the `lightgbm.sklearn` estimators but automatically run a hyperparameter
search using Optuna under the hood.  Any parameters accepted by LightGBM can be
passed to these classes.  Additional parameters control the optimization
process:

- `n_trials`: number of Optuna trials to run.
- `timeout`: maximum optimization time in seconds.
- `cv`: cross‑validation strategy (integer, splitter instance or iterable).
- `param_distributions`: optional Optuna distributions defining a custom search
  space.
- `enable_pruning`: enable Optuna pruning based on early stopping.
- `refit`: refit the best trial after the search finishes.
- `study`: provide an existing Optuna study object.
- `model_dir`: directory where intermediate LightGBM models are stored.

Key attributes exposed after fitting include `best_params_`, `best_score_` and
`booster_` – the underlying LightGBM booster.

The estimators support features such as parallelism via `n_jobs`, early stopping
and group‑aware CV, which are described in the sections below.

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

## Parallelism with `n_jobs`

`n_jobs` controls how many CPU cores are used during both the hyperparameter
search and prediction.  Pass a positive integer to limit the number of threads
or set it to `-1` to use all available cores.

```python
clf = lgb.LGBMClassifier(n_trials=100, n_jobs=-1)
```

During prediction you may also override the value:

```python
y_pred = clf.predict(X_test, n_jobs=2)
```

## Early Stopping

The `fit` method accepts `early_stopping_rounds` which is forwarded to
LightGBM's CV procedure.  When specified, training stops when the validation
score has not improved for the given number of rounds.

```python
clf.fit(X, y, early_stopping_rounds=10)
```

## Group-aware Cross-Validation

If your data contain groups, supply them via the `groups` parameter of `fit`.
OptGBM will use them while performing cross-validation splits.  You can also
provide a `GroupKFold` instance to `cv` for custom splitting strategies.

```python
groups = np.repeat(np.arange(10), len(X) // 10)
cv = GroupKFold(n_splits=5)
clf = lgb.LGBMClassifier(cv=cv)
clf.fit(X, y, groups=groups)
```

## Accessing the Underlying Booster

After fitting, the trained LightGBM booster (or voting ensemble when `refit` is
False) is available via the `booster_` attribute.  You can use this object to
leverage any LightGBM specific functionality.

```python
booster = clf.booster_
importance = booster.feature_importance()
```

## Installation

```
pip install optgbm
```

## Testing

```
tox
```
