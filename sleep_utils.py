"""
sleep_utils.py
Shared functions for sleep analysis
"""

import pandas as pd


def statistics(daily_summary):
    """Calculate sleep duration and quality statistics."""
    
    mean_sleep = daily_summary['sleep_time'].mean()
    std_sleep = daily_summary['sleep_time'].std()
    min_sleep = daily_summary['sleep_time'].min()
    max_sleep = daily_summary['sleep_time'].max()
    median_sleep = daily_summary['sleep_time'].median()
    
    mean_eff = daily_summary['efficiency'].mean()
    std_eff = daily_summary['efficiency'].std()
    
    mean_lat = daily_summary['onset_latency'].mean()
    std_lat = daily_summary['onset_latency'].std()
    
    mean_frag = daily_summary['fragmentation'].mean()
    std_frag = daily_summary['fragmentation'].std()
    
    mean_wake = daily_summary['wake_time'].mean()
    std_wake = daily_summary['wake_time'].std()
    
    mean_intervals = daily_summary['num_sleep_intervals'].mean()
    total_days = len(daily_summary)
    
    # Return as dictionary (no printing for batch mode)
    stats = {
        'total_days': total_days,
        'sleep_time_mean': mean_sleep,
        'sleep_time_std': std_sleep,
        'sleep_time_median': median_sleep,
        'sleep_time_min': min_sleep,
        'sleep_time_max': max_sleep,
        'efficiency_mean': mean_eff,
        'efficiency_std': std_eff,
        'onset_latency_mean': mean_lat,
        'onset_latency_std': std_lat,
        'fragmentation_mean': mean_frag,
        'fragmentation_std': std_frag,
        'wake_time_mean': mean_wake,
        'wake_time_std': std_wake,
        'avg_intervals_per_day': mean_intervals
    }
    
    return stats


def timing_analysis(sleep_df):
    """Analyze sleep timing by category."""
    
    # Convert to datetime
    if 'bedtime' not in sleep_df.columns:
        sleep_df = sleep_df.copy()
        sleep_df['bedtime'] = pd.to_datetime(
            sleep_df['start_date'] + ' ' + sleep_df['start_time']
        )
        sleep_df['waketime'] = pd.to_datetime(
            sleep_df['end_date'] + ' ' + sleep_df['end_time']
        )
    
    # Extract hour as decimal
    sleep_df['bedtime_hour'] = (
        sleep_df['bedtime'].dt.hour + 
        sleep_df['bedtime'].dt.minute / 60
    )
    sleep_df['waketime_hour'] = (
        sleep_df['waketime'].dt.hour + 
        sleep_df['waketime'].dt.minute / 60
    )
    
    # Categorize sleep
    def categorize_sleep(row):
        duration = row['sleep_time']
        start_hour = row['bedtime_hour']
        
        if duration >= 180:
            return "Nighttime Sleep"
        
        if 6 <= start_hour < 12:
            return "Morning Nap"
        elif 12 <= start_hour < 17:
            return "Afternoon Nap"
        elif 17 <= start_hour < 20:
            return "Evening Nap"
        else:
            return "Short Night Sleep"
    
    sleep_df['sleep_category'] = sleep_df.apply(categorize_sleep, axis=1)
    
    # Calculate stats by category
    timing_stats = {}
    
    for category in sleep_df['sleep_category'].unique():
        category_data = sleep_df[sleep_df['sleep_category'] == category]
        
        if len(category_data) == 0:
            continue
        
        timing_stats[category] = {
            'count': len(category_data),
            'start_time_mean': category_data['bedtime_hour'].mean(),
            'start_time_std': category_data['bedtime_hour'].std(),
            'end_time_mean': category_data['waketime_hour'].mean(),
            'end_time_std': category_data['waketime_hour'].std(),
            'duration_mean': category_data['sleep_time'].mean()
        }
    
    return timing_stats


def process_single_subject(file_path):
    """
    Process a single subject file and return all statistics.
    Includes time-of-day analysis (Morning, Afternoon, Evening).
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    dict : Dictionary with all statistics for this subject
    """
    from pathlib import Path
    
    # Extract subject ID from filename
    subject_id = Path(file_path).stem
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Filter to SLEEP only
    sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
    
    # Check if we have data
    if len(sleep_df) == 0:
        return None
    
    # Convert numeric columns
    numeric_columns = ['sleep_time', 'duration', 'efficiency', 'onset_latency', 
                      'fragmentation', 'wake_time']
    for col in numeric_columns:
        if col in sleep_df.columns:
            sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
    
    # Drop rows with missing data
    sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
    
    if len(sleep_df) == 0:
        return None
    
    # Create daily summary
    daily_summary = sleep_df.groupby('start_date').agg({
        'sleep_time': 'sum',
        'duration': 'sum',
        'efficiency': 'mean',
        'onset_latency': 'mean',
        'fragmentation': 'mean',
        'wake_time': 'mean',
        'immobile_time': 'sum',
        'mobile_time': 'sum',
        'interval_number': 'count'
    }).reset_index()
    
    daily_summary.rename(columns={'interval_number': 'num_sleep_intervals'}, inplace=True)
    
    # Get overall statistics
    stats = statistics(daily_summary)
    
    # ⭐ Get time-of-day statistics (Morning, Afternoon, Evening)
    timing_stats = timing_category_analysis(sleep_df)
    
    # Combine into one row
    result = {
        'subject_id': subject_id,
        **stats,
        **timing_stats  # Add timing category stats
    }
    
    return result

def format_time(hour_decimal):
    """Convert decimal hour to readable time format."""
    hours = int(hour_decimal)
    minutes = int((hour_decimal - hours) * 60)
    
    if hours == 0:
        period = "AM"
        display_hour = 12
    elif hours < 12:
        period = "AM"
        display_hour = hours
    elif hours == 12:
        period = "PM"
        display_hour = 12
    else:
        period = "PM"
        display_hour = hours - 12
    
    return f"{display_hour}:{minutes:02d} {period}"


def interpret_consistency(std_minutes):
    """Interpret timing consistency."""
    if std_minutes < 15:
        return "Very Consistent"
    elif std_minutes < 30:
        return "Consistent"
    elif std_minutes < 45:
        return "Moderately Variable"
    elif std_minutes < 60:
        return "Variable"
    else:
        return "Highly Variable"
    
def timing_category_analysis(sleep_df):
    """
    Analyze sleep by time of day categories: Morning, Afternoon, Evening.
    
    Categories:
    - Morning: 10 AM - 11:59 AM
    - Afternoon: 12 PM - 5:59 PM
    - Evening: 6 PM - 9:59 AM (all other times)
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with individual sleep intervals
    
    Returns:
    --------
    dict : Statistics for each time category
    """
    
    # Prepare bedtime and waketime
    if 'bedtime' not in sleep_df.columns:
        sleep_df = sleep_df.copy()
        sleep_df['bedtime'] = pd.to_datetime(
            sleep_df['start_date'] + ' ' + sleep_df['start_time']
        )
        sleep_df['waketime'] = pd.to_datetime(
            sleep_df['end_date'] + ' ' + sleep_df['end_time']
        )
    
    # Extract start hour
    sleep_df['start_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
    
    # Categorize by start time
    def categorize_by_time(start_hour):
        if 10 <= start_hour < 12:
            return "morning"
        elif 12 <= start_hour < 18:
            return "afternoon"
        else:
            return "evening"
    
    sleep_df['time_category'] = sleep_df['start_hour'].apply(categorize_by_time)
    
    # Calculate bedtime and waketime as decimal hours
    sleep_df['bedtime_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
    sleep_df['waketime_hour'] = sleep_df['waketime'].dt.hour + sleep_df['waketime'].dt.minute / 60
    
    results = {}
    
    # Analyze each category
    for category in ['morning', 'afternoon', 'evening']:
        cat_data = sleep_df[sleep_df['time_category'] == category]
        
        if len(cat_data) > 0:
            results[f'{category}_count'] = len(cat_data)
            results[f'{category}_bedtime_mean'] = cat_data['bedtime_hour'].mean()
            results[f'{category}_bedtime_std'] = cat_data['bedtime_hour'].std() if len(cat_data) > 1 else 0
            results[f'{category}_waketime_mean'] = cat_data['waketime_hour'].mean()
            results[f'{category}_waketime_std'] = cat_data['waketime_hour'].std() if len(cat_data) > 1 else 0
            results[f'{category}_duration_mean'] = cat_data['sleep_time'].mean()
            results[f'{category}_duration_std'] = cat_data['sleep_time'].std() if len(cat_data) > 1 else 0
        else:
            # No data for this category
            results[f'{category}_count'] = 0
            results[f'{category}_bedtime_mean'] = None
            results[f'{category}_bedtime_std'] = None
            results[f'{category}_waketime_mean'] = None
            results[f'{category}_waketime_std'] = None
            results[f'{category}_duration_mean'] = None
            results[f'{category}_duration_std'] = None
    
    return results