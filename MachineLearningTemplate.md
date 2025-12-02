```python
from DisplayUtils import display_prediction, display_prediction_season
from sklearn.model_selection import train_test_split
import pandas as pd
```

```python
cf = pd.read_csv("data/CF_FR.csv", index_col="Date", parse_dates=["Date"])
ta = pd.read_csv("data/TA_MEAN_NOTALL.csv", index_col="Date", parse_dates=["Date"])
tp = pd.read_csv("data/TP_FR.csv", index_col="Date", parse_dates=["Date"])
tp_w_opti = pd.read_csv("data/TP_ACC_NOTALL_W_OPTI.csv", index_col="Date", parse_dates=["Date"])
cosSin = pd.read_csv("data/COS_SIN.csv", index_col="Date", parse_dates=["Date"])

X = pd.concat([ta, tp, tp_w_opti, cosSin], axis=1)
y = cf["FR"]
```

```python
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=365, shuffle=False)

# Normalisation
scaler = StandardScaler().set_output(transform="pandas")
columns_to_normalize = X.columns.values.tolist()
columns_to_normalize.remove('sin')
columns_to_normalize.remove('cos')

```

```python
X_train_std = X_train.copy()
X_train_std[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])

X_test_std[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
```

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
y_pred = pd.Series(y_pred, index=y_test.index)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2: {r2:.06f}")
print(f"MSE: {mse:.06f}")
print()

display_prediction(y_test, y_pred)
```

```python

```
