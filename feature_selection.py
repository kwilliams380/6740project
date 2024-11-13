import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with specified file paths
fielding = pd.read_csv('Data/Lahman/Fielding.csv')
batting = pd.read_csv('Data/Lahman/Batting.csv')
pitching = pd.read_csv('Data/Lahman/Pitching.csv')
savant_pitch_data = pd.read_csv('Data/Savant/savant_pitch_data.csv')
players = pd.read_csv('Data/Lahman/People.csv', encoding='latin1')

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
data = pd.merge(data, players[['playerID', 'birthYear']], on='playerID', how='left')

data['age'] = data['yearID'] - data['birthYear']
data.drop(columns=['yearID', 'birthYear'], inplace=True)

# Set correlation threshold (e.g., 0.85)
threshold = 0.85

numeric_data = data.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numeric_data.corr().abs()

# Identify highly correlated features
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

# Drop highly correlated features from the dataset
data = numeric_data.drop(columns=to_drop)

print("Dropped features due to high correlation:", to_drop)

# print(data.columns.tolist())

# # Select only numerical columns for the correlation matrix
# numeric_data = data.select_dtypes(include=[np.number])

# # Calculate the correlation matrix
# correlation_matrix = numeric_data.corr()

# # Plot the correlation matrix as a heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
# plt.title("Correlation Matrix of Numerical Features")
# plt.show()

# Separate pitchers and field players
field_players = data[data.get('HBP_x', pd.Series()).notna()]
pitchers = data[data.get('ERA', pd.Series()).notna()]

# Field Players: Feature Engineering (OBP calculation as a simple proxy for player_value)
for col in ['H_x', 'BB_x', 'HBP_x', 'AB', 'SF_x']:
    if col not in field_players.columns:
        field_players[col] = 0

field_players['OBP'] = (field_players['H_x'] + field_players['BB_x'] + field_players['HBP_x']) / (
    field_players['AB'] + field_players['BB_x'] + field_players['HBP_x'] + field_players['SF_x'])
field_players['OBP'] = field_players['OBP'].fillna(0)
field_players['player_value'] = field_players['OBP']

# Check if 'pitchers' dataset is empty
if pitchers.empty:
    print("No data available for pitchers. Skipping pitcher processing.")
else:
    # Pitchers: Calculate a pitcher-specific metric, e.g., ERA, as player_value (ERA is present as per column listing)
    pitchers['player_value'] = pitchers['ERA'].fillna(0)

    # Processing Pitchers (Directly selecting from the pitchers DataFrame)
    X_pitch = pitchers.drop(columns=['player_value'], errors='ignore').select_dtypes(include=[np.number])
    y_pitch = pitchers['player_value']

    # Impute missing values in X_pitch
    imputer = SimpleImputer(strategy='mean')
    X_pitch_imputed = imputer.fit_transform(X_pitch)

    # Feature Selection
    selector_pitch = SelectKBest(score_func=f_regression, k=15)
    X_pitch_selected = selector_pitch.fit_transform(X_pitch_imputed, y_pitch)
    selected_features_pitch = X_pitch.columns[selector_pitch.get_support(indices=True)]

    # Train-Test Split
    X_train_pitch, X_test_pitch, y_train_pitch, y_test_pitch = train_test_split(X_pitch_selected, y_pitch, test_size=0.2, random_state=42)

    # Model Building and Training
    linear_model_pitch = LinearRegression().fit(X_train_pitch, y_train_pitch)
    random_forest_pitch = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_pitch, y_train_pitch)

    # Predictions and Evaluation
    y_pred_linear_pitch = linear_model_pitch.predict(X_test_pitch)
    y_pred_rf_pitch = random_forest_pitch.predict(X_test_pitch)

    rmse_linear_pitch = mean_squared_error(y_test_pitch, y_pred_linear_pitch, squared=False)
    rmse_rf_pitch = mean_squared_error(y_test_pitch, y_pred_rf_pitch, squared=False)

    # Output Results for Pitchers
    print("\nPitchers:")
    print(f'Selected Features: {selected_features_pitch}')
    print(f'Linear Regression RMSE: {rmse_linear_pitch}')
    print(f'Random Forest RMSE: {rmse_rf_pitch}')

# Processing Field Players
numeric_data_field = field_players.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
X_field = numeric_data_field.drop(columns=['player_value'], errors='ignore')
y_field = numeric_data_field['player_value']
imputer = SimpleImputer(strategy='mean')
X_field_imputed = imputer.fit_transform(X_field)
selector_field = SelectKBest(score_func=f_regression, k=20)
X_field_selected = selector_field.fit_transform(X_field_imputed, y_field)
selected_features_field = X_field.columns[selector_field.get_support(indices=True)]
X_train_field, X_test_field, y_train_field, y_test_field = train_test_split(X_field_selected, y_field, test_size=0.2, random_state=42)
linear_model_field = LinearRegression().fit(X_train_field, y_train_field)
random_forest_field = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_field, y_train_field)
y_pred_linear_field = linear_model_field.predict(X_test_field)
y_pred_rf_field = random_forest_field.predict(X_test_field)
rmse_linear_field = mean_squared_error(y_test_field, y_pred_linear_field, squared=False)
rmse_rf_field = mean_squared_error(y_test_field, y_pred_rf_field, squared=False)

# Output Results for Field Players
print("Field Players:")
print(f'Selected Features: {selected_features_field}')
print(f'Linear Regression RMSE: {rmse_linear_field}')
print(f'Random Forest RMSE: {rmse_rf_field}')
