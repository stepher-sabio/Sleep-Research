"""
debug_subject.py
Debug a specific subject's sleep data
"""

import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_subject.py <subject_csv_file>")
    sys.exit(1)

file_path = sys.argv[1]
subject_id = file_path.split('/')[-1].replace('.csv', '')

print("="*70)
print(f"DEBUGGING: {subject_id}")
print("="*70)

# Load data
df = pd.read_csv(file_path)
print(f"\nTotal rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Filter to SLEEP
sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
print(f"\nSLEEP rows: {len(sleep_df)}")

if len(sleep_df) == 0:
    print("ERROR: No SLEEP data found!")
    sys.exit(1)

# Check sleep_time column
print("\n--- SLEEP_TIME COLUMN ---")
print(f"Data type: {sleep_df['sleep_time'].dtype}")
print(f"Sample values:")
print(sleep_df[['start_date', 'start_time', 'end_time', 'sleep_time']].head(10))

# Convert to numeric
sleep_df['sleep_time'] = pd.to_numeric(sleep_df['sleep_time'], errors='coerce')

print("\n--- AFTER NUMERIC CONVERSION ---")
print(f"Non-null values: {sleep_df['sleep_time'].notna().sum()}")
print(f"Min: {sleep_df['sleep_time'].min()}")
print(f"Max: {sleep_df['sleep_time'].max()}")
print(f"Mean: {sleep_df['sleep_time'].mean():.1f}")

# Categorize
print("\n--- CATEGORIZATION ---")
nighttime = sleep_df[sleep_df['sleep_time'] >= 180]
naps = sleep_df[sleep_df['sleep_time'] < 180]

print(f"Nighttime sleep (≥180 min): {len(nighttime)} periods")
print(f"Naps (<180 min): {len(naps)} periods")

if len(nighttime) > 0:
    print("\nNighttime sleep details:")
    print(nighttime[['start_date', 'start_time', 'end_time', 'sleep_time']])
else:
    print("\n⚠ WARNING: NO NIGHTTIME SLEEP FOUND!")
    print("All sleep periods are <180 minutes")
    print("\nAll sleep periods:")
    print(sleep_df[['start_date', 'start_time', 'end_time', 'sleep_time']].sort_values('sleep_time', ascending=False))

if len(naps) > 0:
    print("\nNap details:")
    print(naps[['start_date', 'start_time', 'end_time', 'sleep_time']])