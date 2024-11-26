import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

fielding = pd.read_csv('Data/Lahman/Fielding.csv')
batting = pd.read_csv('Data/Lahman/Batting.csv')
pitching = pd.read_csv('Data/Lahman/Pitching.csv')
players = pd.read_csv('Data/Lahman/People.csv', encoding='latin1')

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

pitching = pitching[pitching_cols]
batting = batting[batting_cols]
fielding = fielding[fielding_cols]
players = players[['playerID', 'birthYear']]

fielding = (
    fielding.groupby(['playerID', 'yearID', 'teamID', 'stint'])
    .agg('sum')
    .reset_index()
)

position_players = pd.merge(batting, fielding, on=['playerID', 'yearID', 'teamID'], how='outer')
position_players = pd.merge(position_players, players, on='playerID', how='left')
position_players['age'] = position_players['yearID'] - position_players['birthYear']
position_players.drop(columns=['birthYear'], inplace=True)
# Remove catcher-specific rows
position_players = position_players.drop(columns=['PB', 'WP', 'SB_y', 'CS_y', 'ZR'])

pitchers = pd.merge(pitching, players, on='playerID', how='left')
pitchers['age'] = pitchers['yearID'] - pitchers['birthYear']
pitchers.drop(columns=['birthYear'], inplace=True)

if not position_players.empty:
    # Select columns after 1954 - removes tons of NaNs
    position_players_f = position_players[(position_players['yearID'] > 1954)]
    print("Number of rows before filtering:", position_players.shape[0])
    print("Number of rows after filtering:", position_players_f.shape[0])

    X_position = position_players_f.drop(columns=['playerID', 'yearID', 'teamID', 'stint_x', 'stint_y'])
    
    # Fill 0s in rows of players who batted but never fielded
    kept_fielding_cols = ['G_y', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP']
    X_position.loc[X_position[kept_fielding_cols].isna().all(axis=1), kept_fielding_cols] = 0

    scaler = StandardScaler()
    X_position_scaled = scaler.fit_transform(X_position)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_position_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    # print("Cumulative Explained Variance Ratio:", pca.explained_variance_ratio_.cumsum())
    print("Number of components to explain 80% of variance:", (pca.explained_variance_ratio_.cumsum() < 0.8).sum() + 1)
    print("Number of components to explain 90% of variance:", (pca.explained_variance_ratio_.cumsum() < 0.9).sum() + 1)
    print("Number of components to explain 95% of variance:", (pca.explained_variance_ratio_.cumsum() < 0.95).sum() + 1)
    print("Total number of components:", pca.n_components_)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pca_df.corr(), annot=False, cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Selected Features for Position Players")
    plt.show()

    pca_df.to_csv('PositionPlayersPCA.csv', index=False)