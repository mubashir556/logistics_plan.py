import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Simulate Logistics Data
np.random.seed(42)
n_samples = 1000

distance = np.random.uniform(10, 500, n_samples)
weight = np.random.uniform(1, 50, n_samples)
traffic = np.random.uniform(10, 90, n_samples)
stops = np.random.randint(1, 10, n_samples)

# Synthetic target with non-linear relationships and noise
delivery_time = (distance / 45) + (traffic * 0.05) + (stops * 0.4) + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({
    'distance_miles': distance,
    'package_weight_kg': weight,
    'traffic_index': traffic,
    'num_stops': stops,
    'delivery_time_hours': delivery_time
})

# 2. Train-Test Split
X = df[['distance_miles', 'package_weight_kg', 'traffic_index', 'num_stops']]
y = df['delivery_time_hours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Hyperparameter Tuning
rf = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 4. Evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:")
print(f"MAE: {mae:.2f} hours")
print(f"RMSE: {rmse:.2f} hours")
print(f"R-squared: {r2:.4f}")
