"""
batch_visualize.py
Generate visualizations for all subjects
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
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations'

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
    print(f"Processing: {subject_id}...", end=' ')
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
        
        if len(sleep_df) == 0:
            print("⚠ No sleep data")
            failed += 1
            continue
        
        # ⭐ CLEAN DATA: Convert numeric columns and drop invalid rows
        numeric_columns = ['sleep_time', 'duration', 'efficiency', 'onset_latency', 
                          'fragmentation', 'wake_time']
        
        for col in numeric_columns:
            if col in sleep_df.columns:
                sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
        
        # Drop rows with missing critical data
        sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
        
        if len(sleep_df) == 0:
            print("⚠ No valid sleep data after cleaning")
            failed += 1
            continue
        
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
        
        # Drop any rows with NaN in critical columns
        daily_summary = daily_summary.dropna(subset=['sleep_time'])
        
        if len(daily_summary) == 0:
            print("⚠ No valid daily data")
            failed += 1
            continue
        
        # Create subject-specific output folder
        subject_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_folder, exist_ok=True)
        
        # Generate plots
        plot_sleep_raster(sleep_df, subject_id, 
                         save_path=f'{subject_folder}/{subject_id}_raster.png')
        
        plot_sleep_duration(daily_summary, subject_id,
                           save_path=f'{subject_folder}/{subject_id}_duration.png')
        
        plot_sleep_quality(daily_summary, subject_id,
                          save_path=f'{subject_folder}/{subject_id}_quality.png')
        
        plot_timing_consistency(sleep_df, subject_id,
                               save_path=f'{subject_folder}/{subject_id}_timing.png')
        
        print("✓")
        processed += 1
        
    except Exception as e:
        print(f"✗ Error: {e}")
        failed += 1
        # Uncomment for debugging:
        # import traceback
        # traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Successfully processed: {processed} subjects")
print(f"Failed: {failed} subjects")
print(f"\nVisualizations saved to: {output_folder}")
print("="*70)