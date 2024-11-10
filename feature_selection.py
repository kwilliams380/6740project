import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load data with specified file paths
fielding = pd.read_csv('Data/Lahman/Fielding.csv')
batting = pd.read_csv('Data/Lahman/Batting.csv')
pitching = pd.read_csv('Data/Lahman/Pitching.csv')
savant_pitch_data = pd.read_csv('Data/Savant/savant_pitch_data.csv')

# Ensure 'playerID' is of the same type (string) across all dataframes
fielding['playerID'] = fielding['playerID'].astype(str)
batting['playerID'] = batting['playerID'].astype(str)
pitching['playerID'] = pitching['playerID'].astype(str)
savant_pitch_data['player_id'] = savant_pitch_data['player_id'].astype(str)  # Rename to 'playerID' when merging

# Data Preprocessing: Merge datasets
data = pd.merge(fielding, batting, on=['playerID', 'yearID'], how='outer')
data = pd.merge(data, pitching, on=['playerID', 'yearID'], how='outer')
savant_pitch_data.rename(columns={'player_id': 'playerID'}, inplace=True)
data = pd.merge(data, savant_pitch_data, on='playerID', how='left')

# Adjust column names for OBP calculation (use '_x' suffix where necessary)
for col in ['H_x', 'BB_x', 'HBP_x', 'AB', 'SF_x']:
    if col not in data.columns:
        data[col] = 0

# Feature Engineering (OBP calculation as a simple proxy for player_value)
data['OBP'] = (data['H_x'] + data['BB_x'] + data['HBP_x']) / (data['AB'] + data['BB_x'] + data['HBP_x'] + data['SF_x'])
data['OBP'] = data['OBP'].fillna(0)

# Use OBP as a placeholder for 'player_value'
data['player_value'] = data['OBP']

# Select only numeric columns for feature selection and model training
numeric_data = data.select_dtypes(include=[np.number])

# Drop columns that contain only NaN values
numeric_data = numeric_data.dropna(axis=1, how='all')

# Separate features and target variable
X = numeric_data.drop(columns=['player_value'], errors='ignore')
y = numeric_data['player_value']

# Impute missing values with column means
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X_imputed, y)
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Model Building and Training
linear_model = LinearRegression().fit(X_train, y_train)
random_forest = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# Predictions and Evaluation
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = random_forest.predict(X_test)

rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

print(f'Selected Features: {selected_features}')
print(f'Linear Regression RMSE: {rmse_linear}')
print(f'Random Forest RMSE: {rmse_rf}')
