import pandas as pd

# Load the data
file_path = '/Users/stepher/Desktop/Actigraphy2/data_csv/TOSS_102_16mos.csv'
df = pd.read_csv(file_path)


# Filter SLEEP rows only
sleep_df = df[df['interval_type'] == 'SLEEP'].copy()


# Group and aggregate the data
daily_summary = sleep_df.groupby('start_date').agg({
    'sleep_time': 'sum',           # Total sleep per day
    'duration': 'sum',             # Total time in bed
    'efficiency': 'mean',          # Average efficiency
    'onset_latency': 'mean',       # Average latency
    'fragmentation': 'mean',       # Average fragmentation
    'wake_time': 'mean',           # Average wake time
    'immobile_time': 'sum',        # Total immobile
    'mobile_time': 'sum',          # Total mobile
    'interval_number': 'count'     # Number of sleep intervals
}).reset_index()

daily_summary.rename(columns={'interval_number': 'num_sleep_intervals'}, inplace=True)


# ============================================================
# Sleep Quality Analysis Part
# ============================================================
def sleep_quality_analysis(daily_summary):
    mean_sleep = daily_summary['sleep_time'].mean()
    std_sleep = daily_summary['sleep_time'].std()
    min_sleep = daily_summary['sleep_time'].min()
    max_sleep = daily_summary['sleep_time'].max()
    median_sleep = daily_summary['sleep_time'].median()
    
    # Sleep Quality
    mean_eff = daily_summary['efficiency'].mean()
    std_eff = daily_summary['efficiency'].std()
    
    mean_lat = daily_summary['onset_latency'].mean()
    std_lat = daily_summary['onset_latency'].std()
    
    mean_frag = daily_summary['fragmentation'].mean()
    std_frag = daily_summary['fragmentation'].std()
    
    mean_wake = daily_summary['wake_time'].mean()
    std_wake = daily_summary['wake_time'].std()
    
    # Sleep Patterns
    mean_intervals = daily_summary['num_sleep_intervals'].mean()
    total_days = len(daily_summary)
    
    
    # Display Statistics
    
    print("\n" + "="*70)
    print("SLEEP STATISTICS")
    print("="*70)
    
    print(f"\nTotal days analyzed: {total_days}")
    
    print("\n--- SLEEP DURATION ---")
    print(f"  Mean:   {mean_sleep:.1f} minutes/day ({mean_sleep/60:.2f} hours)")
    print(f"  SD:     ± {std_sleep:.1f} minutes")
    print(f"  Median: {median_sleep:.1f} minutes ({median_sleep/60:.2f} hours)")
    print(f"  Range:  {min_sleep:.1f} - {max_sleep:.1f} minutes")
    
    print("\n--- SLEEP QUALITY ---")
    print(f"  Efficiency:       {mean_eff:.1f}% ± {std_eff:.1f}%")
    print(f"  Onset Latency:    {mean_lat:.1f} ± {std_lat:.1f} minutes")
    print(f"  Fragmentation:    {mean_frag:.2f} ± {std_frag:.2f}")
    print(f"  Wake Time (WASO): {mean_wake:.1f} ± {std_wake:.1f} minutes")
    
    print("\n--- SLEEP PATTERNS ---")
    print(f"  Avg sleep intervals/day: {mean_intervals:.1f}")
    
    print("="*70 + "\n")
    

# ============================================================
# Sleep Timing Analysis Part
# ============================================================
        
def sleep_timing_analysis(sleep_df):
    """
    Analyze sleep timing patterns with intelligent categorization.
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with SLEEP intervals
    
    Returns:
    --------
    dict : Dictionary containing timing statistics by sleep type
    """    
    
    # Convert to datetime if not already
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
        
    def categorize_sleep(row):
        """Categorize sleep based on duration and timing."""
        duration = row['sleep_time']
        start_hour = row['bedtime_hour']
        
        # Main sleep: > 3 hours (180 min)
        if duration >= 180:
            return "Nighttime Sleep"
        
        # Naps: categorize by time of day
        if 6 <= start_hour < 12:
            return "Morning Nap"
        elif 12 <= start_hour < 17:
            return "Afternoon Nap"
        elif 17 <= start_hour < 20:
            return "Evening Nap"
        else:
            # Short overnight sleep (unusual but possible)
            return "Short Night Sleep"
    
    sleep_df['sleep_category'] = sleep_df.apply(categorize_sleep, axis=1)
    
    print("\n" + "="*70)
    print("SLEEP TIMING ANALYSIS (BY SLEEP TYPE)")
    print("="*70)
    
    print(f"\nTotal sleep intervals: {len(sleep_df)}")
    print(f"Date range: {sleep_df['start_date'].min()} to {sleep_df['start_date'].max()}")
    
    # Count by category
    category_counts = sleep_df['sleep_category'].value_counts()
    print("\n--- SLEEP TYPE DISTRIBUTION ---")
    for category, count in category_counts.items():
        print(f"  {category}: {count} intervals")
    
    # Stats by category
    timing_stats = {}
    
    for category in sleep_df['sleep_category'].unique():
        category_data = sleep_df[sleep_df['sleep_category'] == category]
        
        if len(category_data) == 0:
            continue
        
        mean_start = category_data['bedtime_hour'].mean()
        std_start = category_data['bedtime_hour'].std()
        mean_end = category_data['waketime_hour'].mean()
        std_end = category_data['waketime_hour'].std()
        mean_duration = category_data['sleep_time'].mean()
        
        print(f"\n--- {category.upper()} ---")
        print(f"  Count: {len(category_data)} intervals")
        print(f"  Avg Start Time:  {format_time(mean_start)} (± {std_start*60:.0f} min)")
        print(f"  Avg End Time:    {format_time(mean_end)} (± {std_end*60:.0f} min)")
        print(f"  Avg Duration:    {mean_duration:.0f} min ({mean_duration/60:.1f} hrs)")
        
        # Consistency
        consistency = interpret_consistency(std_start * 60)
        print(f"  Timing Consistency: {consistency}")
        
        # Store stats
        timing_stats[category] = {
            'count': len(category_data),
            'start_time_mean': mean_start,
            'start_time_std': std_start,
            'end_time_mean': mean_end,
            'end_time_std': std_end,
            'duration_mean': mean_duration,
            'consistency': consistency
        }
    
    print("="*70 + "\n")
    
    return timing_stats

def format_time(hour_decimal):
    """Convert decimal hour to readable time format."""
    hours = int(hour_decimal)
    minutes = int((hour_decimal - hours) * 60)
    
    # Convert to 12-hour format
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

if __name__ == "__main__":
    
    # Test sleep quality analysis
    # sleep_quality_analysis(daily_summary)
    
    # Test sleep timing analysis
    timing_stats = sleep_timing_analysis(sleep_df)
    
    pass