import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load and filter data
batting = pd.read_csv('Batting.csv')
fielding = pd.read_csv('Fielding.csv')
start_year = 2000
batting = batting[batting['yearID'] >= start_year]
fielding = fielding[fielding['yearID'] >= start_year]

# 1. Run Production Component (Offensive Value)
batting['run_production'] = (
    # Direct runs
    batting['R'] +                  # Runs scored
    batting['RBI'] +                # Runs batted in
    -batting['HR'] +                # Subtract HR to avoid double counting
    
    # Base hits (weighted by run expectancy)
    (batting['H'] - batting['2B'] - batting['3B'] - batting['HR']) * 0.5 +  # Singles
    batting['2B'] * 0.8 +          # Doubles
    batting['3B'] * 1.0 +          # Triples
    batting['HR'] * 1.4 +          # Home Runs
    
    # On-base events
    batting['BB'] * 0.3 +          # Walks
    batting['HBP'] * 0.3 +         # Hit By Pitch
    
    # Base running
    batting['SB'] * 0.3 +          # Stolen Bases
    batting['CS'] * -0.6           # Caught Stealing (negative)
)

# 2. Run Prevention Component (Defensive Value)
# Base defensive actions
fielding['base_run_prevention'] = (
    fielding['PO'] * 0.5 +         # Put Outs
    fielding['A'] * 0.4 +          # Assists
    fielding['DP'] * 0.8 +         # Double Plays
    fielding['E'] * -1.0           # Errors (negative)
)

# Position factors with explanations
pos_factors = {
    # Premium defensive positions
    'C':  1.2,   # Catcher: Handles every pitch, controls running game, field general
    'SS': 1.15,  # Shortstop: Most demanding infield position, key defensive anchor
    
    # Important up-the-middle positions
    '2B': 1.1,   # Second Base: Double play pivot, significant range required
    '3B': 1.1,   # Third Base: Hot corner, quick reactions, strong arm needed
    'CF': 1.1,   # Center Field: Controls outfield, most ground to cover
    
    # Corner positions (more offensive-focused)
    '1B': 0.9,   # First Base: Least demanding, mainly receiving throws
    'LF': 0.95,  # Left Field: Corner outfield, less ground to cover
    'RF': 0.95   # Right Field: Corner outfield, stronger arm needed
}

# Apply position adjustments
fielding['pos_factor'] = fielding['POS'].map(pos_factors).fillna(1.0)
fielding['run_prevention'] = fielding['base_run_prevention'] * fielding['pos_factor']

# Aggregate fielding stats by player and year
fielding_agg = fielding.groupby(['playerID', 'yearID']).agg({
    'run_prevention': 'sum',
    'pos_factor': 'mean',
    'POS': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'G': 'sum'
}).reset_index()

# Merge batting and fielding data
player_stats = pd.merge(batting, fielding_agg,
                       on=['playerID', 'yearID'],
                       suffixes=('_bat', '_field'))

# Filter for minimum games
min_games = 50
player_stats = player_stats[player_stats['G_bat'] >= min_games]

# Standardize both components
scaler = StandardScaler()
player_stats['run_production_scaled'] = scaler.fit_transform(player_stats[['run_production']])
player_stats['run_prevention_scaled'] = scaler.fit_transform(player_stats[['run_prevention']])

# Calculate total run value
player_stats['total_run_value'] = (
    0.6 * player_stats['run_production_scaled'] +  # Offensive component (60%)
    0.4 * player_stats['run_prevention_scaled']    # Defensive component (40%)
)

def create_historical_features(data, player_id, target_year, lookback_years=3):
    """
    Create features based on player's historical performance
    """
    historical_data = data[
        (data['playerID'] == player_id) & 
        (data['yearID'] < target_year) & 
        (data['yearID'] >= target_year - lookback_years)
    ]
    
    if len(historical_data) == 0:
        return None
    
    features = {
        'playerID': player_id,
        'target_year': target_year,
        'years_of_data': len(historical_data),
        
        # Average performance metrics
        'avg_run_production': historical_data['run_production'].mean(),
        'avg_run_prevention': historical_data['run_prevention'].mean(),
        'avg_total_value': historical_data['total_run_value'].mean(),
        
        # Trend (year-over-year change)
        'production_trend': historical_data.groupby('yearID')['run_production'].mean().diff().mean(),
        'prevention_trend': historical_data.groupby('yearID')['run_prevention'].mean().diff().mean(),
        
        # Consistency metrics (standard deviation)
        'production_std': historical_data['run_production'].std(),
        'prevention_std': historical_data['run_prevention'].std(),
        
        # Most recent year's performance
        'last_production': historical_data.loc[historical_data['yearID'].idxmax(), 'run_production'],
        'last_prevention': historical_data.loc[historical_data['yearID'].idxmax(), 'run_prevention'],
        
        # Position info
        'primary_position': historical_data['POS'].mode().iloc[0],
        'pos_factor': historical_data['pos_factor'].mean(),
        
        # Games played
        'avg_games': historical_data['G_bat'].mean()
    }
    
    return features

def prepare_prediction_data(player_stats, min_years=3):
    """
    Prepare data for prediction modeling
    """
    training_data = []
    
    for year in sorted(player_stats['yearID'].unique())[min_years:]:
        current_players = player_stats[player_stats['yearID'] == year]['playerID'].unique()
        
        for player in current_players:
            features = create_historical_features(player_stats, player, year)
            
            if features is not None:
                actual = player_stats[
                    (player_stats['playerID'] == player) & 
                    (player_stats['yearID'] == year)
                ]['total_run_value'].iloc[0]
                
                features['actual_value'] = actual
                training_data.append(features)
    
    return pd.DataFrame(training_data)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model, return performance metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    return {
        'model_name': model_name,
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'model': model
    }

# Save the processed player stats
player_stats.to_csv('player_stats_processed.csv', index=False)

# Prepare prediction data
prediction_data = prepare_prediction_data(player_stats)

# Prepare features for modeling
feature_columns = [
    'avg_run_production', 'avg_run_prevention', 'avg_total_value',
    'production_trend', 'prevention_trend',
    'production_std', 'prevention_std',
    'last_production', 'last_prevention',
    'pos_factor', 'avg_games'
]

X = prediction_data[feature_columns]
y = prediction_data['actual_value']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Evaluate all models
results = []
predictions = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results.append(result)
    predictions[name] = result['predictions']

# Create results DataFrame
results_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'R² Score': r['r2_score'],
        'RMSE': r['rmse'],
        'MAE': r['mae'],
        'CV Mean': r['cv_mean'],
        'CV Std': r['cv_std']
    }
    for r in results
])

print("\nModel Comparison:")
print(results_df.round(3))

# Visualize model performance comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R² Score', data=results_df)
plt.xticks(rotation=45)
plt.title('Model Performance Comparison (R² Score)')
plt.tight_layout()
plt.show()

# Visualize predictions vs actual for best model
best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
best_predictions = predictions[best_model_name]

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title(f'Best Model ({best_model_name}) Predictions vs Actual')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance for tree-based models
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.show()

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)

# Save predictions from all models
all_predictions = pd.DataFrame({
    'Actual': y_test,
    **{name: pred for name, pred in predictions.items()}
})
all_predictions.to_csv('all_model_predictions.csv', index=False)

# Save best model
import pickle
best_model = models[best_model_name]
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nBest performing model: {best_model_name}")
print("Results saved to 'model_comparison_results.csv'")
print("Predictions saved to 'all_model_predictions.csv'")
print("Best model saved to 'best_model.pkl'")