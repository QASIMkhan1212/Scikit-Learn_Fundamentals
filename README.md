# Scikit-Learn Fundamentals

A beginner-friendly guide to scikit-learn: the **Core API** (Estimator, Predictor, Transformer) and **Data Preprocessing**, with runnable code.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)

## Contents

- [Part 1: Core API](#part-1-core-api)
- [Part 2: Data Preprocessing](#part-2-data-preprocessing)
- [Golden Rules](#golden-rules)
- [Requirements](#requirements)
- [Next Topics](#next-topics)

---

# Part 1: Core API

## Overview

Every object in scikit-learn follows the same consistent API. Learn this pattern once and it applies to almost every model and preprocessing tool in the library.

| Type | Methods | Job | Example |
|---|---|---|---|
| Estimator | `fit` | Learns from data | All models and transformers |
| Predictor | `fit`, `predict`, `score` | Makes predictions | `LinearRegression`, `LogisticRegression` |
| Transformer | `fit`, `transform`, `fit_transform` | Changes the data | `StandardScaler` |

Run the Core API notebook directly in Google Colab (no setup needed):

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

---

# Part 2: Data Preprocessing

Preprocessing turns raw data into clean numeric input a model can learn from. Almost every preprocessing tool in scikit-learn is a **transformer**.

Run the Preprocessing notebook directly in Google Colab (no setup needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14E-4Foh178PECLJh_kUdSkXMRs9MnZba)

**Standard order:** Split, Impute, Encode, Scale, Model.

## The Five Jobs

| Job | Tools | Use when |
|---|---|---|
| Missing values | `SimpleImputer`, `KNNImputer`, `IterativeImputer` | Data has NaN |
| Encoding | `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder` | Data has text categories |
| Scaling | `StandardScaler`, `MinMaxScaler`, `RobustScaler` | Features have different ranges |
| Transforming | `PowerTransformer`, `QuantileTransformer`, `PolynomialFeatures` | Skewed data, non-linear features |
| Combining | `ColumnTransformer`, `Pipeline` | Mixed column types, safe workflow |

## 1. Missing Values

```python
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="median")            # mean / median / most_frequent / constant
X_train[["age", "salary"]] = num_imp.fit_transform(X_train[["age", "salary"]])
X_test[["age", "salary"]] = num_imp.transform(X_test[["age", "salary"]])

print(num_imp.statistics_)                            # learned fill values
```

- Use `median` when outliers exist, `most_frequent` for categories.
- `X[["col"]]` (double brackets) is needed because transformers expect 2D input.

## 2. Encoding

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Unordered categories (city)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
city_train = ohe.fit_transform(X_train[["city"]])
city_test = ohe.transform(X_test[["city"]])

# Ordered categories (BS < MS < PhD)
oe = OrdinalEncoder(categories=[["BS", "MS", "PhD"]])
edu_train = oe.fit_transform(X_train[["edu"]])
```

- Never use `OrdinalEncoder` on unordered categories: the model would treat one city as greater than another.
- `LabelEncoder` is for the target `y` only.

## 3. Scaling

| Scaler | Formula | Best for |
|---|---|---|
| `StandardScaler` | `(x - mean) / std` | Default choice |
| `MinMaxScaler` | `(x - min) / (max - min)` | Neural networks, bounded range |
| `RobustScaler` | `(x - median) / IQR` | Data with outliers |

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)   # fit on train only
X_test_s = sc.transform(X_test)
```

| Needs scaling | No scaling needed |
|---|---|
| KNN, SVM, K-Means, PCA, linear models | Decision Tree, Random Forest, XGBoost |

## 4. Transforming Distributions

```python
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures

PowerTransformer().fit_transform(X_train[["salary"]])            # reduces skew
PolynomialFeatures(degree=2).fit_transform(X_train[["age"]])     # adds age squared
```

## 5. ColumnTransformer and Pipeline

`Pipeline` chains steps in order. `ColumnTransformer` applies different steps to different columns. Together they apply the fit-on-train rule automatically and prevent data leakage.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])
ord_pipe = Pipeline([
    ("ord", OrdinalEncoder(categories=[["BS", "MS", "PhD"]])),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, ["age", "salary"]),
    ("cat", cat_pipe, ["city"]),
    ("ord", ord_pipe, ["edu"]),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression()),
])

model.fit(X_train, y_train)            # preprocessing is fit on train automatically
print(model.score(X_test, y_test))
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Fitting a scaler on all data before splitting | Split first, or use a `Pipeline` |
| `fit_transform` on the test set | Use `transform` only |
| `LabelEncoder` on features | Use `OrdinalEncoder` or `OneHotEncoder` |
| Passing a 1D Series to a transformer | Use `X[["col"]]` |
| Unseen category crashes at prediction time | `handle_unknown="ignore"` |

---

## Golden Rules

1. Fit on training data only, never on test data.
2. Use `fit_transform` on train and `transform` on test.
3. A trailing underscore (`coef_`) means the value was learned after `fit`.
4. `X` must be 2D `(n_samples, n_features)`; `y` is 1D.
5. Use `Pipeline` to prevent data leakage.

## Requirements

```
numpy
pandas
scikit-learn>=1.2
```

## Next Topics

- [x] Core API: Estimator, Predictor, Transformer
- [x] Preprocessing: imputation, scaling, encoding
- [x] `ColumnTransformer` and `Pipeline`
- [ ] Cross-validation and hyperparameter tuning

## Author

**Muhammad Qasim Khan**
GitHub: [QASIMkhan1212](https://github.com/QASIMkhan1212)
