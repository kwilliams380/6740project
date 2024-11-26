import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

fielding = pd.read_csv('Data/Lahman/Fielding.csv')
batting = pd.read_csv('Data/Lahman/Batting.csv')
pitching = pd.read_csv('Data/Lahman/Pitching.csv')
players = pd.read_csv('Data/Lahman/People.csv', encoding='latin1')
salaries = pd.read_csv('Data/Lahman/Salaries.csv')

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
position_players = pd.merge(position_players, salaries.drop('lgID',axis=1), on=['playerID', 'yearID', 'teamID'], how='left')
position_players = pd.merge(position_players, players, on='playerID', how='left')
position_players['age'] = position_players['yearID'] - position_players['birthYear']
position_players.drop(columns=['birthYear'], inplace=True)
# Remove catcher-specific rows
position_players = position_players.drop(columns=['PB', 'WP', 'SB_y', 'CS_y', 'ZR'])

pitchers = pd.merge(pitching, players, on='playerID', how='left')
pitchers = pd.merge(pitchers, salaries, on=['playerID', 'yearID', 'teamID'], how='left')
pitchers['age'] = pitchers['yearID'] - pitchers['birthYear']
pitchers.drop(columns=['birthYear'], inplace=True)

if not position_players.empty:
    ## PREPROCESSING
    # Select columns between 1985 and 2016 - years with salary data
    position_players_f = position_players[(position_players['yearID'] >= 1985) & (position_players['yearID'] <= 2016)]
    print("Number of rows before filtering:", position_players.shape[0])
    print("Number of rows after filtering:", position_players_f.shape[0])

    print("Number of columns with NaN value in 'salary' column:", position_players_f['salary'].isnull().sum())
    position_players_f = position_players_f.dropna(subset=['salary'])

    X_position = position_players_f.drop(columns=['playerID', 'yearID', 'teamID', 'salary', 'stint_x', 'stint_y'])
    y_position = position_players_f['salary']

    # Fill 0s in rows of players who batted but never fielded
    kept_fielding_cols = ['G_y', 'GS', 'InnOuts', 'PO', 'A', 'E', 'DP']
    X_position.loc[X_position[kept_fielding_cols].isna().all(axis=1), kept_fielding_cols] = 0

    scaler = StandardScaler()
    X_position_scaled = scaler.fit_transform(X_position)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_position_scaled)

    print("Final number of rows: ", X_pca.shape[0])
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_position, test_size=0.2, random_state=0)

    hyperparam_grid = {
        'n_estimators': [50, 75, 100, 125, 150, 175, 200, 300, 400, 500],
        'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    }

    model = RandomForestRegressor(random_state=0)
    grid_search = GridSearchCV(model, hyperparam_grid, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)



# VIF / Correlation - PCA to fix? Try to reduce. Ben has VIF at ~30
# Dollar valuation of player - Built model to predict salary
  # Ideally, same structure of inputs as survival analysis
  # Predict next year's salary
  # 


