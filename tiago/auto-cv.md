---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from custom_estimator import HydroEstimator

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
```

```python
cf = pd.read_csv('data/CF_1d.csv', index_col='Date', parse_dates=True)
ta = pd.read_csv('data/TA_1d.csv', index_col='Date', parse_dates=True)
tp = pd.read_csv('data/TP_1d.csv', index_col='Date', parse_dates=True)
fr_columns = ta.columns[ta.columns.str.startswith('FR')]
print(fr_columns)
ta = tp[fr_columns]
tp = tp[fr_columns]
```

```python
cf.head(10)
```

```python
data = pd.DataFrame()
data['CF'] = cf['FR']
data = data.merge(ta, left_on='Date', right_on='Date')
data = data.merge(tp, left_on='Date', right_on='Date', suffixes=['_TA', '_TP'])
data
```

```python
from sklearn.metrics import r2_score, mean_squared_error

def display_result(y_true, y_pred):
    """Affiche les résultats de prédiction / réels."""
    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot 1
    ax1.set_title("Capacity factor predictions")
    ax1.plot(y_true.index, y_true, color="tab:blue", label="Actual")
    ax1.plot(y_true.index, y_pred, color="tab:red", label="Predicted")

    ax1.set_xlim(y_true.index[0], y_true.index[-1])
    ax1.legend(loc="lower right", title="Capacity Factor")

    # Plot 2
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.scatter(y_true, y_pred, color="tab:blue", s=10)

    left, right = ax2.get_xlim()
    bottom, top = ax2.get_ylim()
    lb = min(left, bottom) - 0.01
    ub = max(right, top) + 0.01
    ax2.set_ylim(lb, ub)
    ax2.set_xlim(lb, ub)
    ax2.axline((lb, lb), (ub, ub), color="tab:red")

    plt.show()
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# Utilisation corrigée
X = data.drop(columns=['CF'])
y = data['CF']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=365, shuffle=False)
X
```

```python
import pandas as pd
import scipy.stats.distributions as dists
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

max_lag, max_window = 180, 180

params = {
    "w_10": dists.randint(5, max_window),
    "w_C1": dists.randint(5, max_window), "w_C2": dists.randint(5, max_window),
    "w_D1": dists.randint(5, max_window), "w_D2": dists.randint(5, max_window),
    "w_E1": dists.randint(5, max_window), "w_E2": dists.randint(5, max_window),
    "w_F1": dists.randint(5, max_window), "w_F2": dists.randint(5, max_window), "w_F3": dists.randint(5, max_window),
    "w_G0": dists.randint(5, max_window), "w_H0": dists.randint(5, max_window),
    "w_I1": dists.randint(5, max_window), "w_I2": dists.randint(5, max_window), "w_I3": dists.randint(5, max_window),
    "w_J1": dists.randint(5, max_window), "w_J2": dists.randint(5, max_window),
    "w_K1": dists.randint(5, max_window), "w_K2": dists.randint(5, max_window),
    "w_L0": dists.randint(5, max_window),
    "l_10": dists.randint(5, max_lag), "l_B0": dists.randint(5, max_lag),
    "l_C1": dists.randint(5, max_lag), "l_C2": dists.randint(5, max_lag),
    "l_D1": dists.randint(5, max_lag), "l_D2": dists.randint(5, max_lag),
    "l_E1": dists.randint(5, max_lag), "l_E2": dists.randint(5, max_lag),
    "l_F1": dists.randint(5, max_lag), "l_F2": dists.randint(5, max_lag), "l_F3": dists.randint(5, max_lag),
    "l_G0": dists.randint(5, max_lag), "l_H0": dists.randint(5, max_lag),
    "l_I1": dists.randint(5, max_lag), "l_I2": dists.randint(5, max_lag), "l_I3": dists.randint(5, max_lag),
    "l_J1": dists.randint(5, max_lag), "l_J2": dists.randint(5, max_lag),
    "l_K1": dists.randint(5, max_lag), "l_K2": dists.randint(5, max_lag),
    "l_L0": dists.randint(5, max_lag)
}

reg = HydroEstimator(test_size=365, model=Ridge())


cv = RandomizedSearchCV(
    estimator=reg, 
    param_distributions=params, 
    n_iter=50, 
    cv=3, 
    scoring='r2', 
    n_jobs=-1, 
    random_state=42
)

cv.fit(X_train, y_train)

print(f"Meilleurs paramètres: {cv.best_params_}")
print(f"Meilleur score: {cv.best_score_:.4f}")

y_pred = cv.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")

display_result(y_test, y_pred)
```

```python

```

```python

```
