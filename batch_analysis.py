"""
batch_analysis.py
Process all subjects and create summary CSV with time-of-day statistics
"""

import pandas as pd
from pathlib import Path
import os
from sleep_utils import process_single_subject, format_time

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_file = '/Users/stepher/Desktop/Actigraphy2/results/all_subjects_summary.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print("="*70)
print("BATCH ANALYSIS - ALL SUBJECTS")
print("="*70)
print(f"Input folder: {input_folder}")
print(f"Output file: {output_file}")
print("="*70 + "\n")

# Process all files
all_results = []
processed = 0
failed = 0

for file_path in Path(input_folder).glob('*.csv'):
    subject_id = file_path.stem
    print(f"Processing {subject_id}...", end=' ')
    
    try:
        result = process_single_subject(file_path)
        
        if result is not None:
            all_results.append(result)
            print("✓")
            processed += 1
        else:
            print("⚠ No valid data")
            failed += 1
    
    except Exception as e:
        print(f"✗ Error: {e}")
        failed += 1

# Create DataFrame
if all_results:
    summary_df = pd.DataFrame(all_results)
    
    # ⭐ Convert time columns to clock format
    time_columns = [
        'evening_bedtime_mean', 'evening_waketime_mean',
        'morning_bedtime_mean', 'morning_waketime_mean',
        'afternoon_bedtime_mean', 'afternoon_waketime_mean'
    ]
    
    for col in time_columns:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: format_time(x) if pd.notna(x) else None
            )
    
    # ⭐ Convert std columns from decimal hours to minutes
    std_columns = [
        'evening_bedtime_std', 'evening_waketime_std',
        'morning_bedtime_std', 'morning_waketime_std',
        'afternoon_bedtime_std', 'afternoon_waketime_std'
    ]
    
    for col in std_columns:
        if col in summary_df.columns:
            # Convert from hours to minutes
            summary_df[col] = summary_df[col] * 60
            # Round to 1 decimal place
            summary_df[col] = summary_df[col].round(1)
    
    # Rename std columns to indicate minutes
    rename_dict = {}
    for col in std_columns:
        if col in summary_df.columns:
            new_col = col.replace('_std', '_std_minutes')
            rename_dict[col] = new_col
    
    summary_df.rename(columns=rename_dict, inplace=True)
    
    # ⭐ Reorder columns for better readability
    # Start with subject ID and overall stats
    column_order = ['subject_id', 'total_days']
    
    # Overall sleep statistics
    overall_cols = [col for col in summary_df.columns 
                   if col.startswith('sleep_time_') or 
                      col.startswith('efficiency_') or
                      col.startswith('onset_latency_') or
                      col.startswith('fragmentation_') or
                      col.startswith('wake_time_') or
                      col == 'avg_intervals_per_day']
    column_order.extend([c for c in overall_cols if c in summary_df.columns])
    
    # Morning statistics
    morning_cols = [col for col in summary_df.columns if col.startswith('morning_')]
    column_order.extend(sorted(morning_cols))
    
    # Afternoon statistics
    afternoon_cols = [col for col in summary_df.columns if col.startswith('afternoon_')]
    column_order.extend(sorted(afternoon_cols))
    
    # Evening statistics
    evening_cols = [col for col in summary_df.columns if col.startswith('evening_')]
    column_order.extend(sorted(evening_cols))
    
    # Reorder
    summary_df = summary_df[column_order]
    
    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully processed: {processed} subjects")
    print(f"Failed: {failed} subjects")
    print(f"\nOutput saved to: {output_file}")
    print("="*70)
    
    # Print sample of time-of-day statistics
    print("\n" + "="*70)
    print("SAMPLE TIME-OF-DAY STATISTICS")
    print("="*70)
    
    if processed > 0:
        sample = summary_df.iloc[0]
        print(f"\nSubject: {sample['subject_id']}")
        print("\n--- MORNING (10 AM - 11:59 AM) ---")
        print(f"  Count: {sample.get('morning_count', 0)}")
        if sample.get('morning_count', 0) > 0:
            print(f"  Bedtime: {sample.get('morning_bedtime_mean', 'N/A')}")
            print(f"  Wake Time: {sample.get('morning_waketime_mean', 'N/A')}")
            print(f"  Duration: {sample.get('morning_duration_mean', 0):.1f} ± {sample.get('morning_duration_std', 0):.1f} min")
        
        print("\n--- AFTERNOON (12 PM - 5:59 PM) ---")
        print(f"  Count: {sample.get('afternoon_count', 0)}")
        if sample.get('afternoon_count', 0) > 0:
            print(f"  Bedtime: {sample.get('afternoon_bedtime_mean', 'N/A')}")
            print(f"  Wake Time: {sample.get('afternoon_waketime_mean', 'N/A')}")
            print(f"  Duration: {sample.get('afternoon_duration_mean', 0):.1f} ± {sample.get('afternoon_duration_std', 0):.1f} min")
        
        print("\n--- EVENING (6 PM - 9:59 AM) ---")
        print(f"  Count: {sample.get('evening_count', 0)}")
        if sample.get('evening_count', 0) > 0:
            print(f"  Bedtime: {sample.get('evening_bedtime_mean', 'N/A')}")
            print(f"  Wake Time: {sample.get('evening_waketime_mean', 'N/A')}")
            print(f"  Duration: {sample.get('evening_duration_mean', 0):.1f} ± {sample.get('evening_duration_std', 0):.1f} min")
        
        print("="*70)

else:
    print("\n⚠ No subjects were successfully processed")