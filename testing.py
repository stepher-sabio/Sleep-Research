import pandas as pd

# Load the data
file_path = '/Users/stepher/Desktop/Actigraphy2/data_csv/TOSS_103_16mos.csv'
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


# Data Analysis Part
def statistics(daily_summary):
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
  

if __name__ == "__main__":
    
    # Check if the data is loaded correctly
    # print("Original data shape:", df.shape)
    # print("\nFirst few rows:")
    # print(df.head())
    
    # Check if the data is filtered correctly
    # print(f"\nFiltered to SLEEP only: {len(sleep_df)} rows")
    # print("\nSleep data:")
    # print(sleep_df)
    
    # Check if the data is aggregated correctly
    statistics(daily_summary)
    
    pass