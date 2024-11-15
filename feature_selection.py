import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

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
savant_pitch_data['player_id'] = savant_pitch_data['player_id'].astype(str)

# Rename overlapping columns before merging
batting.rename(columns=lambda x: f"{x}_batting" if x in fielding.columns and x != "playerID" and x != "yearID" else x, inplace=True)
pitching.rename(columns=lambda x: f"{x}_pitching" if x in fielding.columns and x != "playerID" and x != "yearID" else x, inplace=True)

# Merge datasets
data = pd.merge(fielding, batting, on=['playerID', 'yearID'], how='outer')
data = pd.merge(data, pitching, on=['playerID', 'yearID'], how='outer')
savant_pitch_data.rename(columns={'player_id': 'playerID'}, inplace=True)
data = pd.merge(data, savant_pitch_data, on='playerID', how='left')
data = pd.merge(data, players[['playerID', 'birthYear']], on='playerID', how='left')

# Resolve duplicate columns by merging "_x" and "_y" suffixes
for col in data.columns:
    if col.endswith('_x'):
        base_col = col[:-2]  # Remove "_x" suffix
        if f'{base_col}_y' in data.columns:
            # Combine values, prioritize "_x", and drop "_y"
            data[base_col] = data[col].fillna(data[f'{base_col}_y'])
            data.drop(columns=[col, f'{base_col}_y'], inplace=True)

# Add age column
data['age'] = data['yearID'] - data['birthYear']
data.drop(columns=['yearID', 'birthYear'], inplace=True)

# Separate pitchers and field players without imputing missing values
field_players = data[data.get('HBP', pd.Series()).notna()].copy()
pitchers = data[data.get('ERA', pd.Series()).notna()].copy()

# Feature Selection for Field Players
if not field_players.empty:
    X_field = field_players.select_dtypes(include=[np.number]).dropna(axis=1, how='any')  # Drop columns with any NaNs

    # Perform feature selection with a random target
    selector_field = SelectKBest(score_func=f_regression, k='all')
    selector_field.fit(X_field, np.random.rand(len(X_field)))  # Use random target as placeholder
    selected_features_field = X_field.columns[selector_field.get_support(indices=True)]

    print("\nField Players - Selected Features:")
    print(selected_features_field)

    # Plot correlation matrix of selected features for field players
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_field[selected_features_field].corr(), annot=False, cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Selected Features for Field Players")
    plt.show()

# Feature Selection for Pitchers
if not pitchers.empty:
    X_pitch = pitchers.select_dtypes(include=[np.number]).dropna(axis=1, how='any')  # Drop columns with any NaNs

    # Perform feature selection with a random target
    selector_pitch = SelectKBest(score_func=f_regression, k='all')
    selector_pitch.fit(X_pitch, np.random.rand(len(X_pitch)))  # Use random target as placeholder
    selected_features_pitch = X_pitch.columns[selector_pitch.get_support(indices=True)]

    print("\nPitchers - Selected Features:")
    print(selected_features_pitch)

    # Plot correlation matrix of selected features for pitchers
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_pitch[selected_features_pitch].corr(), annot=False, cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Selected Features for Pitchers")
    plt.show()
