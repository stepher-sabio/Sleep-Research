"""
check_weekend_data.py
Phase 1: Check if we have sufficient weekday/weekend data for social jet lag analysis

Checks:
1. How many subjects have weekend data?
2. How many subjects have adequate weekday + weekend coverage?
3. Day-of-week distribution
4. Date range per subject
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'

print("="*70)
print("WEEKDAY/WEEKEND DATA CHECK")
print("="*70)
print()

# ============================================================================
# LOAD AND ANALYZE DATA
# ============================================================================

results = []
total_files = 0
files_with_weekend = 0
files_with_sufficient_data = 0

for file_path in Path(input_folder).glob('*.csv'):
    subject_id = file_path.stem
    total_files += 1
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Filter to SLEEP
        sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
        
        if len(sleep_df) == 0:
            continue
        
        # Parse dates
        sleep_df['date'] = pd.to_datetime(sleep_df['start_date'])
        sleep_df['day_of_week'] = sleep_df['date'].dt.day_name()
        sleep_df['day_of_week_num'] = sleep_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Classify weekday vs weekend
        # Weekend = Saturday (5) and Sunday (6)
        sleep_df['is_weekend'] = sleep_df['day_of_week_num'].isin([5, 6])
        
        # Count unique dates
        unique_dates = sleep_df['date'].unique()
        n_days = len(unique_dates)
        
        # Count weekday vs weekend days
        weekday_dates = sleep_df[~sleep_df['is_weekend']]['date'].unique()
        weekend_dates = sleep_df[sleep_df['is_weekend']]['date'].unique()
        
        n_weekdays = len(weekday_dates)
        n_weekends = len(weekend_dates)
        
        # Day-of-week distribution
        dow_counts = sleep_df['day_of_week'].value_counts().to_dict()
        
        # Date range
        min_date = sleep_df['date'].min()
        max_date = sleep_df['date'].max()
        date_span = (max_date - min_date).days
        
        # Check if sufficient data
        # Criteria: At least 3 weekdays AND at least 1 weekend day
        has_weekend = n_weekends >= 1
        has_sufficient = (n_weekdays >= 3) and (n_weekends >= 1)
        
        if has_weekend:
            files_with_weekend += 1
        if has_sufficient:
            files_with_sufficient_data += 1
        
        # Extract age
        try:
            parts = subject_id.split('_')
            age_part = [p for p in parts if 'mos' in p.lower()]
            age = int(age_part[0].replace('mos', '').replace('mo', '')) if age_part else None
        except:
            age = None
        
        results.append({
            'subject_id': subject_id,
            'age_months': age,
            'total_days': n_days,
            'weekdays': n_weekdays,
            'weekends': n_weekends,
            'has_weekend': has_weekend,
            'has_sufficient': has_sufficient,
            'date_span_days': date_span,
            'min_date': min_date.strftime('%Y-%m-%d'),
            'max_date': max_date.strftime('%Y-%m-%d'),
            'monday': dow_counts.get('Monday', 0),
            'tuesday': dow_counts.get('Tuesday', 0),
            'wednesday': dow_counts.get('Wednesday', 0),
            'thursday': dow_counts.get('Thursday', 0),
            'friday': dow_counts.get('Friday', 0),
            'saturday': dow_counts.get('Saturday', 0),
            'sunday': dow_counts.get('Sunday', 0)
        })
        
    except Exception as e:
        print(f"  ⚠ Error processing {subject_id}: {e}")
        continue

# ============================================================================
# SUMMARY REPORT
# ============================================================================

results_df = pd.DataFrame(results)

print("="*70)
print("DATA COVERAGE SUMMARY")
print("="*70)
print(f"\nTotal subjects processed: {total_files}")
print(f"Subjects with valid sleep data: {len(results_df)}")
print(f"Subjects with ANY weekend data: {files_with_weekend} ({files_with_weekend/len(results_df)*100:.1f}%)")
print(f"Subjects with SUFFICIENT data (≥3 weekdays + ≥1 weekend): {files_with_sufficient_data} ({files_with_sufficient_data/len(results_df)*100:.1f}%)")

print("\n" + "="*70)
print("SAMPLE SIZE BREAKDOWN")
print("="*70)

print("\nBy weekday count:")
print(results_df['weekdays'].value_counts().sort_index().to_string())

print("\nBy weekend count:")
print(results_df['weekends'].value_counts().sort_index().to_string())

print("\n" + "="*70)
print("DAY-OF-WEEK DISTRIBUTION (across all subjects)")
print("="*70)

dow_totals = {
    'Monday': results_df['monday'].sum(),
    'Tuesday': results_df['tuesday'].sum(),
    'Wednesday': results_df['wednesday'].sum(),
    'Thursday': results_df['thursday'].sum(),
    'Friday': results_df['friday'].sum(),
    'Saturday': results_df['saturday'].sum(),
    'Sunday': results_df['sunday'].sum()
}

for day, count in dow_totals.items():
    print(f"  {day:12s}: {count:4d} days")

print(f"\n  Total weekdays: {sum([v for k,v in dow_totals.items() if k not in ['Saturday', 'Sunday']])} days")
print(f"  Total weekends: {dow_totals['Saturday'] + dow_totals['Sunday']} days")

# ============================================================================
# SUBJECTS WITH SUFFICIENT DATA
# ============================================================================

print("\n" + "="*70)
print("SUBJECTS WITH SUFFICIENT DATA FOR SOCIAL JET LAG ANALYSIS")
print("="*70)

sufficient_df = results_df[results_df['has_sufficient']].copy()

if len(sufficient_df) > 0:
    sufficient_df = sufficient_df.sort_values(['age_months', 'subject_id'])
    
    print(f"\nTotal: {len(sufficient_df)} subjects")
    print(f"\nBy age group:")
    if 'age_months' in sufficient_df.columns:
        age_counts = sufficient_df['age_months'].value_counts().sort_index()
        for age, count in age_counts.items():
            print(f"  {age} months: {count} subjects")
    
    print(f"\nDetailed breakdown:")
    print(f"{'Subject ID':<25} {'Age':<6} {'Weekdays':<10} {'Weekends':<10} {'Total':<8} {'Span':<10}")
    print("-"*70)
    
    for _, row in sufficient_df.head(20).iterrows():  # Show first 20
        print(f"{row['subject_id']:<25} {row['age_months'] if pd.notna(row['age_months']) else 'N/A':<6} "
              f"{row['weekdays']:<10} {row['weekends']:<10} {row['total_days']:<8} {row['date_span_days']:<10}")
    
    if len(sufficient_df) > 20:
        print(f"... and {len(sufficient_df) - 20} more subjects")

else:
    print("\n⚠ WARNING: NO SUBJECTS HAVE SUFFICIENT DATA FOR ANALYSIS")
    print("\nMost common limitation:")
    print(f"  - Subjects with 0 weekend days: {len(results_df[results_df['weekends'] == 0])}")
    print(f"  - Subjects with <3 weekdays: {len(results_df[results_df['weekdays'] < 3])}")

# ============================================================================
# SAVE DETAILED REPORT
# ============================================================================

output_file = '/Users/stepher/Desktop/Actigraphy2/results/weekend_data_check.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Detailed report saved to: {output_file}")

# ============================================================================
# EXAMPLE DATA INSPECTION
# ============================================================================

if len(sufficient_df) > 0:
    print("\n" + "="*70)
    print("EXAMPLE: DATA STRUCTURE FOR ONE SUBJECT")
    print("="*70)
    
    # Pick subject with good coverage
    example_subject = sufficient_df.iloc[0]['subject_id']
    print(f"\nSubject: {example_subject}")
    
    # Load and show structure
    example_file = Path(input_folder) / f"{example_subject}.csv"
    example_df = pd.read_csv(example_file)
    example_sleep = example_df[example_df['interval_type'] == 'SLEEP'].copy()
    
    example_sleep['date'] = pd.to_datetime(example_sleep['start_date'])
    example_sleep['day_of_week'] = example_sleep['date'].dt.day_name()
    
    print("\nSleep periods by day:")
    print(f"{'Date':<12} {'Day of Week':<12} {'Start Time':<12} {'End Time':<12} {'Duration (min)':<15}")
    print("-"*70)
    
    for _, row in example_sleep.head(10).iterrows():
        print(f"{row['start_date']:<12} {row['day_of_week']:<12} {row['start_time']:<12} "
              f"{row['end_time']:<12} {row['sleep_time']:<15}")
    
    if len(example_sleep) > 10:
        print(f"... and {len(example_sleep) - 10} more sleep periods")

# ============================================================================
# RECOMMENDATION
# ============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if files_with_sufficient_data >= 10:
    print("\n✅ PROCEED WITH SOCIAL JET LAG ANALYSIS")
    print(f"\nYou have {files_with_sufficient_data} subjects with adequate weekday/weekend coverage.")
    print("This is sufficient for meaningful statistical analysis.")
    
    print("\nNext steps:")
    print("  1. Run full social jet lag analysis")
    print("  2. Compare weekday vs weekend patterns")
    print("  3. Calculate individual SJL scores")
    print("  4. Analyze Monday recovery effects")
    print("  5. Test age group differences")
    
    print("\nExpected outputs:")
    print("  - 3 comprehensive figures")
    print("  - 4 CSV files with metrics")
    print("  - Statistical test results")

elif files_with_weekend >= 5:
    print("\n⚠ LIMITED DATA - PROCEED WITH CAUTION")
    print(f"\nYou have {files_with_weekend} subjects with some weekend data,")
    print(f"but only {files_with_sufficient_data} with adequate coverage.")
    
    print("\nOptions:")
    print("  A. Proceed with reduced sample (statistical power may be limited)")
    print("  B. Relax criteria (e.g., 2 weekdays + 1 weekend)")
    print("  C. Focus on descriptive analysis only (no hypothesis testing)")

else:
    print("\n❌ INSUFFICIENT DATA FOR SOCIAL JET LAG ANALYSIS")
    print(f"\nOnly {files_with_weekend} subjects have ANY weekend data.")
    
    print("\nPossible reasons:")
    print("  - Data collection periods too short")
    print("  - Data collection avoided weekends")
    print("  - Dataset is weekday-only by design")
    
    print("\nAlternative analyses:")
    print("  - Day-of-week effects within weekdays only")
    print("  - Monday vs Friday comparisons")
    print("  - Individual variability analysis")

print("\n" + "="*70)