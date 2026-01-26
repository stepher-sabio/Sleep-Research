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
    """
    from pathlib import Path
    
    subject_id = Path(file_path).stem
    
    # Load data
    df = pd.read_csv(file_path)
    sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
    
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
    
    # Get statistics
    stats = statistics(daily_summary)
    timing_stats = timing_analysis(sleep_df)
    
    # Combine into one row
    result = {
        'subject_id': subject_id,
        **stats
    }
    
    # Add timing stats for nighttime sleep (if exists)
    if 'Nighttime Sleep' in timing_stats:
        night = timing_stats['Nighttime Sleep']
        result['nighttime_count'] = night['count']
        
        # ⭐ CONVERT TO CLOCK TIME
        result['nighttime_bedtime_mean'] = format_time(night['start_time_mean'])
        result['nighttime_bedtime_std_minutes'] = round(night['start_time_std'] * 60, 1)
        
        result['nighttime_waketime_mean'] = format_time(night['end_time_mean'])
        result['nighttime_waketime_std_minutes'] = round(night['end_time_std'] * 60, 1)
        
        result['nighttime_duration_mean'] = night['duration_mean']
    
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