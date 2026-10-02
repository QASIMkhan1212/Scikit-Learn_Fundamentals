# scikit-learn Core API

A beginner-friendly guide to the three building blocks of scikit-learn: **Estimator**, **Predictor** and **Transformer**, with runnable code.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)

## Overview

Every object in scikit-learn follows the same consistent API. Learn this pattern once and it applies to almost every model and preprocessing tool in the library.

| Type | Methods | Job | Example |
|---|---|---|---|
| Estimator | `fit` | Learns from data | All models and transformers |
| Predictor | `fit`, `predict`, `score` | Makes predictions | `LinearRegression`, `LogisticRegression` |
| Transformer | `fit`, `transform`, `fit_transform` | Changes the data | `StandardScaler` |


Or run it directly in Google Colab (no setup needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fcx4B9XFHd2YH954_mv7jEO17p4niyMU?usp=sharing)

## Concepts and Code

### 1. Estimator

An object that **learns from data** using `fit()`.

- Hyperparameters are set in the constructor.
- Attributes learned from data end with an underscore (`coef_`, `mean_`).

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)        # learned after fit
print(model.intercept_)
```

### 2. Predictor

An estimator that can **make predictions**. It adds `predict()` and `score()`. Classifiers also provide `predict_proba()`.

```python
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))     # R2 for regressors, accuracy for classifiers
```

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
clf.predict(X_test)          # class labels
clf.predict_proba(X_test)    # probability per class
```

### 3. Transformer

An estimator that **changes the data**. It adds `transform()` and `fit_transform()`.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)                     # learn mean and std
X_train_s = sc.transform(X_train)   # apply
X_test_s = sc.transform(X_test)

# shortcut for training data only
X_train_s = sc.fit_transform(X_train)
```

## Standard Workflow

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()                    # Transformer
X_train_s = sc.fit_transform(X_train)    # fit on train only
X_test_s = sc.transform(X_test)          # transform only

clf = LogisticRegression(max_iter=200)   # Predictor
clf.fit(X_train_s, y_train)
print("Accuracy:", clf.score(X_test_s, y_test))
```

## Data Leakage: What Not To Do

```python
# Correct: learn statistics from train, reuse them on test
sc = StandardScaler().fit(X_train)
X_test_s = sc.transform(X_test)

# Wrong: re-learns statistics from the test set
X_test_s = StandardScaler().fit_transform(X_test)
```

Fitting on test data lets information from the test set leak into training and gives misleadingly high scores.

## Golden Rules

1. Fit on training data only, never on test data.
2. Use `fit_transform` on train and `transform` on test.
3. A trailing underscore (`coef_`) means the value was learned after `fit`.
4. `X` must be 2D `(n_samples, n_features)`; `y` is 1D.

## Requirements

```
numpy
scikit-learn>=1.2
```

## Next Topics

- [ ] Preprocessing: imputation, scaling, encoding
- [ ] `ColumnTransformer` and `Pipeline`
- [ ] Cross-validation and hyperparameter tuning

## Author

**Muhammad Qasim Khan**
GitHub: [QASIMkhan1212](https://github.com/QASIMkhan1212)
