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

# Rename 'player_id' in Savant data to match 'playerID'
savant_pitch_data.rename(columns={'player_id': 'playerID'}, inplace=True)

# Filter relevant columns for each dataset
# Pitching-specific columns
pitching_cols = [
    'playerID', 'yearID', 'teamID', 'stint', 'W', 'L', 'G', 'GS', 'CG', 'SHO', 'SV', 
    'IPouts', 'H', 'ER', 'HR', 'BB', 'SO', 'BAOpp', 'ERA', 'WP', 'BK', 'BFP', 'GF'
]

# Combine batting and fielding columns for position players
batting_cols = [
    'playerID', 'yearID', 'teamID', 'stint', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 
    'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP'
]
fielding_cols = [
    'playerID', 'yearID', 'teamID', 'stint', 'POS', 'G', 'GS', 'InnOuts', 'PO', 'A', 
    'E', 'DP', 'PB', 'WP', 'SB', 'CS', 'ZR'
]

# Filter each dataset to keep only relevant columns
pitching = pitching[pitching_cols]
batting = batting[[col for col in batting_cols if col in batting.columns]]
fielding = fielding[[col for col in fielding_cols if col in fielding.columns]]

# Merge batting and fielding data to form a combined position player dataset
position_players = pd.merge(batting, fielding, on=['playerID', 'yearID', 'teamID', 'stint', 'G'], how='outer')

# Add birth year for age calculation to both datasets
players = players[['playerID', 'birthYear']]

# Merge position player stats with player info to calculate age
position_players = pd.merge(position_players, players, on='playerID', how='left')
position_players['age'] = position_players['yearID'] - position_players['birthYear']
position_players.drop(columns=['birthYear'], inplace=True)

position_players.to_csv('position_players_dataframe.csv')

# Merge pitching stats with player info to calculate age
pitchers = pd.merge(pitching, players, on='playerID', how='left')
pitchers['age'] = pitchers['yearID'] - pitchers['birthYear']
pitchers.drop(columns=['birthYear'], inplace=True)

pitchers.to_csv('pitchers_dataframe.csv')

# Feature Selection for Position Players
if not position_players.empty:
    # Fill NaN values with 0 or mean to avoid dropping columns
    X_pos = position_players.select_dtypes(include=[np.number]).fillna(0)  # Fills NaNs with 0

    # Perform feature selection with a random target
    selector_pos = SelectKBest(score_func=f_regression, k='all')
    selector_pos.fit(X_pos, np.random.rand(len(X_pos)))  # Use random target as placeholder
    selected_features_pos = X_pos.columns[selector_pos.get_support(indices=True)]

    print("\nPosition Players - Selected Features:")
    print(selected_features_pos)

    # Plot correlation matrix of selected features for position players
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_pos[selected_features_pos].corr(), annot=False, cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Selected Features for Position Players")
    plt.show()


# Feature Selection for Pitchers
if not pitchers.empty:
    # Fill NaN values with 0 to avoid dropping columns
    X_pitch = pitchers.select_dtypes(include=[np.number]).fillna(0)

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
