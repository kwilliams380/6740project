import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
# position_players = pd.merge(position_players, players, on='playerID', how='left')
# position_players['age'] = position_players['yearID'] - position_players['birthYear']
# position_players.drop(columns=['birthYear'], inplace=True)
# Remove catcher-specific rows
position_players = position_players.drop(columns=['PB', 'WP', 'SB_y', 'CS_y', 'ZR'])

# pitchers = pd.merge(pitching, players, on='playerID', how='left')
pitchers = pd.merge(pitching, salaries, on=['playerID', 'yearID', 'teamID'], how='left')
# pitchers['age'] = pitchers['yearID'] - pitchers['birthYear']
# pitchers.drop(columns=['birthYear'], inplace=True)

if not position_players.empty:
    def prepare_salary_prediction_dataset(df):
        df = df.sort_values(['playerID', 'yearID'])
        
        # Create a list to store valid rows
        valid_rows = []
        
        # Group by player
        for _, player_group in df.groupby('playerID'):
            # Iterate through player's seasons
            for i in range(1, len(player_group)):
                # Previous season's stats
                prev_stats = player_group.iloc[i-1]
                # Current season's salary
                current_salary = player_group.iloc[i]['salary']
                
                # Create a row with previous year's stats and current year's salary
                row = prev_stats.copy()
                row['target_salary'] = current_salary
                valid_rows.append(row)
        
        # Convert to DataFrame
        prepared_df = pd.DataFrame(valid_rows)
        
        return prepared_df

    # Prepare the dataset
    position_players_f = position_players[(position_players['yearID'] >= 1985) & (position_players['yearID'] <= 2016)]
    position_players_f = position_players_f.dropna(subset=['salary'])

    # Create dataset where each row is previous year's stats and current year's salary
    salary_prediction_df = prepare_salary_prediction_dataset(position_players_f)

    # Select features and target
    X = salary_prediction_df.drop(columns=['playerID', 'yearID', 'teamID', 'salary', 'target_salary', 'stint_x', 'stint_y'])
    y = salary_prediction_df['target_salary']

    # Split and scale
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
    }

    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                               param_grid=param_grid,
                               cv=5,
                               scoring='r2',
                               n_jobs=-1,
                               return_train_score=True)
    
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print("Optimal number of trees:", grid_search.best_params_['n_estimators'])

    # Extract results
    cv_results = grid_search.cv_results_
    mean_test_r2 = cv_results['mean_test_score']
    mean_train_r2 = cv_results['mean_train_score']
    n_estimators = param_grid['n_estimators']

    # Create a plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=n_estimators, y=mean_test_r2, label="Test R^2", marker='o')
    sns.lineplot(x=n_estimators, y=mean_train_r2, label="Train R^2", marker='o')

    plt.title("R^2 vs. Number of Trees (Position Players)", fontsize=16)
    plt.xlabel("Number of Trees", fontsize=14)
    plt.ylabel("R^2", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

if not pitchers.empty:
    def prepare_salary_prediction_dataset(df):
        df = df.sort_values(['playerID', 'yearID'])
        
        # Create a list to store valid rows
        valid_rows = []
        
        # Group by player
        for _, player_group in df.groupby('playerID'):
            # Iterate through player's seasons
            for i in range(1, len(player_group)):
                # Previous season's stats
                prev_stats = player_group.iloc[i-1]
                # Current season's salary
                current_salary = player_group.iloc[i]['salary']
                
                # Create a row with previous year's stats and current year's salary
                row = prev_stats.copy()
                row['target_salary'] = current_salary
                valid_rows.append(row)
        
        # Convert to DataFrame
        prepared_df = pd.DataFrame(valid_rows)
        
        return prepared_df

    # Prepare the dataset
    pitchers_f = pitchers[(pitchers['yearID'] >= 1985) & (pitchers['yearID'] <= 2016)]
    pitchers_f = pitchers_f.dropna(subset=['salary'])

    # Create dataset where each row is previous year's stats and current year's salary
    salary_prediction_df = prepare_salary_prediction_dataset(pitchers_f)

    # Select features and target
    X = salary_prediction_df.drop(columns=['playerID', 'yearID', 'teamID', 'salary', 'target_salary', 'stint_x', 'stint_y'])
    y = salary_prediction_df['target_salary']

    # Split and scale
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
    }

    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                               param_grid=param_grid,
                               cv=5,
                               scoring='r2',
                               n_jobs=-1,
                               return_train_score=True)
    
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print("Optimal number of trees:", grid_search.best_params_['n_estimators'])

    # Extract results
    cv_results = grid_search.cv_results_
    mean_test_r2 = cv_results['mean_test_score']
    mean_train_r2 = cv_results['mean_train_score']
    n_estimators = param_grid['n_estimators']

    # Create a plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=n_estimators, y=mean_test_r2, label="Test R^2", marker='o')
    sns.lineplot(x=n_estimators, y=mean_train_r2, label="Train R^2", marker='o')

    plt.title("R^2 vs. Number of Trees (Pitchers)", fontsize=16)
    plt.xlabel("Number of Trees", fontsize=14)
    plt.ylabel("R^2", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()