"""
batch_analysis.py
Process all subject files and create summary CSV
"""

import pandas as pd
from pathlib import Path
import os
from sleep_utils import process_single_subject

# ============================================================
# CONFIGURATION
# ============================================================

input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/results'
output_file = 'all_subjects_summary.csv'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# ============================================================
# PROCESS ALL FILES
# ============================================================

print("="*70)
print("BATCH SLEEP ANALYSIS")
print("="*70)
print(f"Input folder: {input_folder}")
print(f"Output: {output_folder}/{output_file}")
print("="*70 + "\n")

all_results = []
processed = 0
failed = 0

# Loop through all CSV files
for file_path in Path(input_folder).glob('*.csv'):
    
    try:
        print(f"Processing: {file_path.name}...", end=' ')
        
        # Process the subject
        result = process_single_subject(str(file_path))
        
        if result is not None:
            all_results.append(result)
            print("✓")
            processed += 1
        else:
            print("⚠ No sleep data")
            failed += 1
            
    except Exception as e:
        print(f"✗ Error: {e}")
        failed += 1

# ============================================================
# CREATE SUMMARY DATAFRAME
# ============================================================

if len(all_results) > 0:
    summary_df = pd.DataFrame(all_results)
    
    # Sort by subject ID
    summary_df = summary_df.sort_values('subject_id')
    
    # Save to CSV
    output_path = os.path.join(output_folder, output_file)
    summary_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully processed: {processed} subjects")
    print(f"Failed: {failed} subjects")
    print(f"\nOutput saved to: {output_path}")
    print("="*70)
    
    # Display first few rows
    print("\nPreview:")
    print(summary_df.head())
    
else:
    print("\n⚠ No subjects were successfully processed!")