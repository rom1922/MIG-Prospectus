```python
from MachineLearningDataset import MLHydroDataset
from DisplayUtils import display_prediction, display_prediction_season
from sklearn.model_selection import train_test_split
import pandas as pd
```

```python
dataset = MLHydroDataset(data_directory="data", correlation_file="correlation_features_hyperparameters.csv")
dataset.prepare_features()
X = dataset.X
y = dataset.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=365, shuffle=False)

X_train_std = dataset.fit_transform(X_train)
X_test_std = dataset.transform(X_test)
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

display_prediction_season(y_test, y_pred)
```

```python

```
