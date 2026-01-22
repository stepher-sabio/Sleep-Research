"""
analyze_single.py
Analyze a single subject with detailed output
"""

import pandas as pd
from sleep_utils import statistics, timing_analysis, format_time, interpret_consistency

# Load the data
file_path = '/Users/stepher/Desktop/Actigraphy2/data_csv/TOSS_103_16mos.csv'
df = pd.read_csv(file_path)

# Filter SLEEP rows only
sleep_df = df[df['interval_type'] == 'SLEEP'].copy()

# Group and aggregate
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

if __name__ == "__main__":
    
    # Get stats (returns dict, no printing)
    stats = statistics(daily_summary)
    
    # Print them nicely
    print("\n" + "="*70)
    print("SLEEP STATISTICS")
    print("="*70)
    print(f"\nTotal days: {stats['total_days']}")
    print(f"Mean sleep: {stats['sleep_time_mean']:.1f} min ({stats['sleep_time_mean']/60:.2f} hrs)")
    print(f"Efficiency: {stats['efficiency_mean']:.1f}%")
    # ... etc
    
    # Timing analysis
    timing_stats = timing_analysis(sleep_df)