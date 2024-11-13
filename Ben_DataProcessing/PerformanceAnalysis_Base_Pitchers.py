import pandas as pd
import numpy as np
from scipy import stats

def load_and_preprocess_data(filepath, min_pitches=500):
    """
    Load and preprocess the Statcast pitching data.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing pitch data
    min_pitches : int
        Minimum total pitches required to be included in analysis (default=500)
    """
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Filter for minimum total pitches
    pitcher_totals = df.groupby('player_name')['total_pitches'].first()
    qualified_pitchers = pitcher_totals[pitcher_totals >= min_pitches].index
    df = df[df['player_name'].isin(qualified_pitchers)]
    
    # Handle any missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Remove any rows with all missing values
    df = df.dropna(how='all')
    
    print(f"\nAnalyzing {len(qualified_pitchers)} pitchers with {min_pitches}+ total pitches")
    
    return df

def apply_reliability_adjustment(score, total_pitches, baseline_pitches=1000):
    """
    Apply a reliability adjustment to scores based on sample size.
    Scores from pitchers with fewer pitches are regressed toward the mean.
    
    Parameters:
    -----------
    score : float
        The original score (0-100)
    total_pitches : int
        Number of pitches thrown
    baseline_pitches : int
        Number of pitches considered "fully reliable" (default=1000)
    
    Returns:
    --------
    float
        Adjusted score
    """
    reliability_factor = min(1.0, total_pitches / baseline_pitches)
    # Regress to mean (50) based on sample size
    return score * reliability_factor + 50 * (1 - reliability_factor)

def calculate_stuff_score(pitcher_data):
    """
    Calculate a "stuff" score based on velocity, spin rate, extension and movement.
    """
    # Weight each pitch type by its usage percentage
    weighted_data = pitcher_data.copy()
    for col in ['velocity', 'spin_rate', 'release_extension', 'effective_speed']:
        weighted_data[col] = weighted_data[col] * weighted_data['pitch_percent']
    
    pitcher_stats = weighted_data.groupby('player_name').agg({
        'velocity': 'sum',
        'spin_rate': 'sum',
        'release_extension': 'sum',
        'effective_speed': 'sum',
        'launch_speed': 'mean',
        'total_pitches': 'first'
    }).reset_index()
    
    # Calculate z-scores for each metric
    for col in ['velocity', 'spin_rate', 'release_extension', 'effective_speed']:
        mean = pitcher_stats[col].mean()
        std = pitcher_stats[col].std()
        pitcher_stats[f'{col}_zscore'] = (pitcher_stats[col] - mean) / std
    
    # Launch speed is better when lower
    mean_launch = pitcher_stats['launch_speed'].mean()
    std_launch = pitcher_stats['launch_speed'].std()
    pitcher_stats['launch_speed_zscore'] = -1 * (pitcher_stats['launch_speed'] - mean_launch) / std_launch
    
    # Calculate overall stuff score
    pitcher_stats['stuff_score'] = (
        0.30 * pitcher_stats['velocity_zscore'] +
        0.25 * pitcher_stats['spin_rate_zscore'] +
        0.20 * pitcher_stats['release_extension_zscore'] +
        0.15 * pitcher_stats['effective_speed_zscore'] +
        0.10 * pitcher_stats['launch_speed_zscore']
    )
    
    # Convert to 0-100 scale
    pitcher_stats['stuff_score'] = ((pitcher_stats['stuff_score'] - 
                                   pitcher_stats['stuff_score'].min()) /
                                  (pitcher_stats['stuff_score'].max() - 
                                   pitcher_stats['stuff_score'].min()) * 100)
    
    # Apply reliability adjustment
    pitcher_stats['stuff_score'] = pitcher_stats.apply(
        lambda x: apply_reliability_adjustment(x['stuff_score'], x['total_pitches']),
        axis=1
    )
    
    return pitcher_stats[['player_name', 'stuff_score', 'total_pitches']]

def calculate_command_score(pitcher_data):
    """
    Calculate a "command" score based on control metrics.
    """
    weighted_data = pitcher_data.copy()
    for col in ['xba', 'xwoba', 'takes']:
        weighted_data[col] = weighted_data[col] * weighted_data['pitch_percent']
    
    pitcher_stats = weighted_data.groupby('player_name').agg({
        'takes': 'sum',
        'xba': 'sum',
        'xwoba': 'sum',
        'launch_angle': 'mean',
        'total_pitches': 'first'
    }).reset_index()
    
    # Calculate take rate
    pitcher_stats['take_rate'] = pitcher_stats['takes'] / pitcher_stats['total_pitches']
    
    # Calculate z-scores
    for col in ['take_rate', 'launch_angle']:
        mean = pitcher_stats[col].mean()
        std = pitcher_stats[col].std()
        pitcher_stats[f'{col}_zscore'] = (pitcher_stats[col] - mean) / std
    
    # xBA and xwOBA are better when lower for pitchers
    for col in ['xba', 'xwoba']:
        mean = pitcher_stats[col].mean()
        std = pitcher_stats[col].std()
        pitcher_stats[f'{col}_zscore'] = -1 * (pitcher_stats[col] - mean) / std
    
    # Calculate command score
    pitcher_stats['command_score'] = (
        0.35 * pitcher_stats['xba_zscore'] +
        0.35 * pitcher_stats['xwoba_zscore'] +
        0.15 * pitcher_stats['take_rate_zscore'] +
        0.15 * pitcher_stats['launch_angle_zscore']
    )
    
    # Convert to 0-100 scale
    pitcher_stats['command_score'] = ((pitcher_stats['command_score'] - 
                                     pitcher_stats['command_score'].min()) /
                                    (pitcher_stats['command_score'].max() - 
                                    pitcher_stats['command_score'].min()) * 100)
    
    # Apply reliability adjustment
    pitcher_stats['command_score'] = pitcher_stats.apply(
        lambda x: apply_reliability_adjustment(x['command_score'], x['total_pitches']),
        axis=1
    )
    
    return pitcher_stats[['player_name', 'command_score', 'total_pitches']]

def calculate_outcome_score(pitcher_data):
    """
    Calculate an "outcome" score based on results metrics.
    """
    weighted_data = pitcher_data.copy()
    for col in ['woba', 'ba', 'slg', 'iso']:
        weighted_data[col] = weighted_data[col] * weighted_data['pitch_percent']
    
    pitcher_stats = weighted_data.groupby('player_name').agg({
        'woba': 'sum',
        'ba': 'sum',
        'slg': 'sum',
        'iso': 'sum',
        'pitcher_run_exp': 'sum',
        'total_pitches': 'first'
    }).reset_index()
    
    # Normalize pitcher run expectancy per pitch
    pitcher_stats['run_exp_per_pitch'] = pitcher_stats['pitcher_run_exp'] / pitcher_stats['total_pitches']
    
    # Calculate z-scores (negative for most since lower is better)
    for col in ['woba', 'ba', 'slg', 'iso']:
        mean = pitcher_stats[col].mean()
        std = pitcher_stats[col].std()
        pitcher_stats[f'{col}_zscore'] = -1 * (pitcher_stats[col] - mean) / std
    
    # Run expectancy is already in the right direction (positive is better)
    mean_run = pitcher_stats['run_exp_per_pitch'].mean()
    std_run = pitcher_stats['run_exp_per_pitch'].std()
    pitcher_stats['run_exp_zscore'] = (pitcher_stats['run_exp_per_pitch'] - mean_run) / std_run
    
    # Calculate outcome score
    pitcher_stats['outcome_score'] = (
        0.30 * pitcher_stats['woba_zscore'] +
        0.20 * pitcher_stats['ba_zscore'] +
        0.20 * pitcher_stats['slg_zscore'] +
        0.15 * pitcher_stats['iso_zscore'] +
        0.15 * pitcher_stats['run_exp_zscore']
    )
    
    # Convert to 0-100 scale
    pitcher_stats['outcome_score'] = ((pitcher_stats['outcome_score'] - 
                                     pitcher_stats['outcome_score'].min()) /
                                    (pitcher_stats['outcome_score'].max() - 
                                    pitcher_stats['outcome_score'].min()) * 100)
    
    # Apply reliability adjustment
    pitcher_stats['outcome_score'] = pitcher_stats.apply(
        lambda x: apply_reliability_adjustment(x['outcome_score'], x['total_pitches']),
        axis=1
    )
    
    return pitcher_stats[['player_name', 'outcome_score', 'total_pitches']]

def print_pitcher_stats(report, pitcher_name):
    """
    Print detailed stats for a specific pitcher
    """
    overall = report['overall_scores']
    
    print(f"\nDetailed Statistics for {pitcher_name}")
    print("=" * 80)
    
    pitcher_scores = overall[overall['player_name'] == pitcher_name]
    if not pitcher_scores.empty:
        print("\nOverall Scores:")
        print(f"Total Pitches:     {pitcher_scores.iloc[0]['total_pitches']:,d}")
        print(f"Overall Score:     {pitcher_scores.iloc[0]['overall_score']:.1f}")
        print(f"Stuff Score:       {pitcher_scores.iloc[0]['stuff_score']:.1f}")
        print(f"Command Score:     {pitcher_scores.iloc[0]['command_score']:.1f}")
        print(f"Outcome Score:     {pitcher_scores.iloc[0]['outcome_score']:.1f}")
        
        print("\nPercentile Rankings:")
        for score in ['overall_score', 'stuff_score', 'command_score', 'outcome_score']:
            percentile = stats.percentileofscore(overall[score], pitcher_scores.iloc[0][score])
            print(f"{score.replace('_', ' ').title():15} {percentile:.1f}")
        
        # Add reliability indication
        reliability = min(100, pitcher_scores.iloc[0]['total_pitches'] / 1000 * 100)
        print(f"\nReliability Score: {reliability:.1f}%")

def print_top_pitchers(overall_scores, n=10):
    """
    Print the top N pitchers and their scores in a formatted table
    """
    print(f"\nTop {n} Pitchers Overall:")
    print("-" * 100)
    print(f"{'Rank':<6}{'Name':<30}{'Pitches':<10}{'Overall':<10}{'Stuff':<10}{'Command':<10}{'Outcome':<10}")
    print("-" * 100)
    
    top_n = overall_scores.head(n)
    for i, row in enumerate(top_n.itertuples(), 1):
        print(f"{i:<6}{row.player_name:<30}{row.total_pitches:<10,d}{row.overall_score:<10.1f}"
              f"{row.stuff_score:<10.1f}{row.command_score:<10.1f}{row.outcome_score:<10.1f}")

def calculate_overall_score(stuff_scores, command_scores, outcome_scores):
    """
    Calculate an overall pitcher score combining stuff, command, and outcomes.
    """
    overall = stuff_scores.merge(command_scores, on=['player_name', 'total_pitches'])
    overall = overall.merge(outcome_scores, on=['player_name', 'total_pitches'])
    
    # Calculate overall score with weights
    overall['overall_score'] = (
        0.35 * overall['stuff_score'] +
        0.25 * overall['command_score'] +
        0.40 * overall['outcome_score']
    )
    
    # Round all scores to 2 decimal places
    score_columns = ['stuff_score', 'command_score', 'outcome_score', 'overall_score']
    overall[score_columns] = overall[score_columns].round(2)
    
    return overall.sort_values('overall_score', ascending=False)

def generate_comprehensive_report(filepath, min_pitches=500):
    """
    Generate a comprehensive pitching analysis report.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing pitch data
    min_pitches : int
        Minimum number of pitches required for inclusion (default=500)
    """
    # Load and preprocess data
    data = load_and_preprocess_data(filepath, min_pitches)
    
    # Calculate all scores
    stuff_scores = calculate_stuff_score(data)
    command_scores = calculate_command_score(data)
    outcome_scores = calculate_outcome_score(data)
    overall_scores = calculate_overall_score(stuff_scores, command_scores, outcome_scores)
    
    # Combine into report
    report = {
        'overall_scores': overall_scores
    }
    
    # Print summary statistics
    print_top_pitchers(overall_scores)
    
    return report

# Example usage:
report = generate_comprehensive_report('savant_pitch_data.csv', min_pitches=500)
print_pitcher_stats(report, "Cole, Gerrit")