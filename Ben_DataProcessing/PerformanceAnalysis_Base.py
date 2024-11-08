import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Analysis outputs
recent_year = player_stats['yearID'].max()
print(f"\nTop Players for {recent_year} by Position:")
for pos in pos_factors.keys():
    pos_players = player_stats[
        (player_stats['yearID'] == recent_year) & 
        (player_stats['POS'] == pos)
    ].sort_values('total_run_value', ascending=False).head(3)
    
    if not pos_players.empty:
        print(f"\n{pos} (Position Factor: {pos_factors[pos]}):")
        for _, player in pos_players.iterrows():
            print(f"Player: {player['playerID']}")
            print(f"- Run Production: {player['run_production']:.1f}")
            print(f"- Run Prevention: {player['run_prevention']:.1f}")
            print(f"- Total Value: {player['total_run_value']:.2f}")

# Visualizations
plt.figure(figsize=(12, 6))
sns.scatterplot(data=player_stats[player_stats['yearID'] == recent_year],
                x='run_production_scaled',
                y='run_prevention_scaled',
                hue='POS',
                size='total_run_value',
                sizes=(50, 400),
                alpha=0.6)

plt.title('Run Production vs Prevention by Position')
plt.xlabel('Offensive Value (Standardized)')
plt.ylabel('Defensive Value (Standardized)')
plt.grid(True, alpha=0.3)
plt.show()

# Position value distributions
plt.figure(figsize=(12, 6))
pos_values = player_stats.groupby('POS').agg({
    'run_production': 'mean',
    'run_prevention': 'mean',
    'total_run_value': 'mean'
}).round(3)

print("\nAverage Values by Position:")
print(pos_values.sort_values('total_run_value', ascending=False))

# Show correlation between components
correlation_matrix = player_stats[[
    'run_production', 'run_prevention', 'total_run_value'
]].corr()

print("\nCorrelation between Components:")
print(correlation_matrix.round(3))

player_stats.to_csv('player_stats_processed.csv', index=False)