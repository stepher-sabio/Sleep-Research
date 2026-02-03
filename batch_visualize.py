"""
batch_visualize.py
Generate visualizations for all subjects - SIMPLIFIED with DEBUG
"""

import pandas as pd
from pathlib import Path
import os
from visualize_sleep import (
    plot_sleep_raster,
    plot_sleep_duration,
    plot_sleep_quality,
    plot_timing_consistency
)

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/individuals'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("BATCH VISUALIZATION GENERATION")
print("="*70)
print(f"Input: {input_folder}")
print(f"Output: {output_folder}")
print("="*70 + "\n")

processed = 0
failed = 0

# Process each file
for file_path in Path(input_folder).glob('*.csv'):
    
    subject_id = file_path.stem
    print(f"\nProcessing: {subject_id}...")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
        
        if len(sleep_df) == 0:
            print("  ⚠ No sleep data")
            failed += 1
            continue
        
        print(f"  Found {len(sleep_df)} sleep intervals")
        
        # Clean data
        numeric_columns = ['sleep_time', 'duration', 'efficiency', 'onset_latency', 
                          'fragmentation', 'wake_time']
        
        for col in numeric_columns:
            if col in sleep_df.columns:
                sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
        
        sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
        
        if len(sleep_df) == 0:
            print("  ⚠ No valid sleep data")
            failed += 1
            continue
        
        print(f"  After cleaning: {len(sleep_df)} valid intervals")
        
        # ⭐ DEBUG: Show sleep times
        if 'bedtime' not in sleep_df.columns:
            sleep_df['bedtime'] = pd.to_datetime(
                sleep_df['start_date'] + ' ' + sleep_df['start_time']
            )
            sleep_df['waketime'] = pd.to_datetime(
                sleep_df['end_date'] + ' ' + sleep_df['end_time']
            )
        
        print("  Sleep periods:")
        for _, row in sleep_df.iterrows():
            print(f"    {row['start_date']}: {row['start_time']} - {row['end_time']} ({row['sleep_time']:.0f} min)")
        
        # Create daily summary
        daily_summary = sleep_df.groupby('start_date').agg({
            'sleep_time': 'sum',
            'duration': 'sum',
            'efficiency': 'mean',
            'onset_latency': 'mean',
            'fragmentation': 'mean',
            'wake_time': 'mean',
            'interval_number': 'count'
        }).reset_index()
        
        daily_summary.rename(columns={'interval_number': 'num_sleep_intervals'}, inplace=True)
        daily_summary = daily_summary.dropna(subset=['sleep_time'])
        
        if len(daily_summary) == 0:
            print("  ⚠ No valid daily data")
            failed += 1
            continue
        
        # Create subject-specific output folder
        subject_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_folder, exist_ok=True)
        
        # Generate plots
        print("  Generating plots...", end=' ')
        
        plot_sleep_raster(sleep_df, subject_id, 
                         save_path=f'{subject_folder}/{subject_id}_raster.png')
        
        plot_sleep_duration(daily_summary, subject_id,
                           save_path=f'{subject_folder}/{subject_id}_duration.png')
        
        plot_sleep_quality(daily_summary, subject_id,
                          save_path=f'{subject_folder}/{subject_id}_quality.png')
        
        plot_timing_consistency(sleep_df, subject_id,
                               save_path=f'{subject_folder}/{subject_id}_timing.png')
        
        print("✓")
        print(f"  ✓ {subject_id} completed")
        processed += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Successfully processed: {processed} subjects")
print(f"Failed: {failed} subjects")
print(f"\nVisualizations saved to: {output_folder}")
print("="*70)