import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm



# Load 164x31 matrix from .npy or similar format
# Each element: [athletes, adv_athletes, num_sports, host_flag, avg_prev3, real_medals]
data_matrix = np.load('country_year_data.npy', allow_pickle=True)

n_countries, n_years = data_matrix.shape
features = []
targets = []

# Extract training data for USA (assumed index 0)
usa_index = 0  # Update if different
for year in range(n_years):
    row = data_matrix[usa_index][year]
    if row is not None and len(row) == 6:
        x = row[:5]  # 5 features
        y = row[5]   # target: real medals
        features.append(x)
        targets.append(y)

X_train = np.array(features)
y_train = np.array(targets)



# Define parameter grid
param_grid = {
    'n_estimators': list(range(50, 301, 50)),
    'max_depth': list(range(5, 31, 5))
}

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")



# Store prediction results
prediction_results = []

# Loop through every country and year
for i in tqdm(range(n_countries)):
    for j in range(n_years):
        row = data_matrix[i][j]
        if row is not None and len(row) == 6:
            x = np.array(row[:5]).reshape(1, -1)
            true_y = row[5]

            # Get all tree predictions (for interval estimation)
            all_tree_preds = np.array([tree.predict(x)[0] for tree in best_model.estimators_])

            # Mean prediction
            mean_pred = np.mean(all_tree_preds)

            # Compute percentiles for 38%, 60%, 90% confidence intervals
            lower_38, upper_38 = np.percentile(all_tree_preds, [31, 69])
            lower_60, upper_60 = np.percentile(all_tree_preds, [20, 80])
            lower_90, upper_90 = np.percentile(all_tree_preds, [5, 95])

            prediction_results.append({
                'Country_Index': i,
                'Year_Index': j,
                'True_Medals': true_y,
                'Predicted_Medals': mean_pred,
                'CI_38_Lower': lower_38,
                'CI_38_Upper': upper_38,
                'CI_60_Lower': lower_60,
                'CI_60_Upper': upper_60,
                'CI_90_Lower': lower_90,
                'CI_90_Upper': upper_90
            })



result_df = pd.DataFrame(prediction_results)
result_df.to_csv('medal_prediction_with_confidence.csv', index=False)

