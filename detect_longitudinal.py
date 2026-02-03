"""
detect_longitudinal.py
Check if dataset contains longitudinal data (same subjects at multiple ages)
"""

import pandas as pd

# Load data
input_csv = '/Users/stepher/Desktop/Actigraphy2/results/all_subjects_summary.csv'
df = pd.read_csv(input_csv)

# Extract subject ID (without age) and age
def extract_subject_and_age(subject_id):
    """
    Extract base subject ID and age
    e.g., 'TOSS_102_16mos' → ('TOSS_102', 16)
    """
    try:
        parts = subject_id.split('_')
        # Find age part
        age_part = [p for p in parts if 'mos' in p.lower()]
        if age_part:
            age_str = age_part[0].replace('mos', '').replace('mo', '')
            age = int(age_str)
            # Base ID is everything before age
            base_id = '_'.join([p for p in parts if 'mos' not in p.lower()])
            return base_id, age
        return None, None
    except:
        return None, None

# Apply extraction
df[['base_subject_id', 'age_months']] = df['subject_id'].apply(
    lambda x: pd.Series(extract_subject_and_age(x))
)

# Remove rows without valid data
df = df[df['base_subject_id'].notna()]

print("="*70)
print("LONGITUDINAL DATA DETECTION")
print("="*70)

# Count observations per subject
subject_counts = df['base_subject_id'].value_counts()

# Identify longitudinal subjects (>1 timepoint)
longitudinal_subjects = subject_counts[subject_counts > 1]

print(f"\nTotal unique subjects: {len(subject_counts)}")
print(f"Subjects with multiple timepoints: {len(longitudinal_subjects)}")
print(f"Subjects with single timepoint: {len(subject_counts[subject_counts == 1])}")

if len(longitudinal_subjects) > 0:
    print("\n" + "="*70)
    print("LONGITUDINAL SUBJECTS FOUND!")
    print("="*70)
    
    print(f"\nSubjects tracked across development:")
    for subject, count in longitudinal_subjects.items():
        ages = df[df['base_subject_id'] == subject]['age_months'].sort_values().tolist()
        print(f"  {subject}: {count} timepoints at ages {ages} months")
    
    # Save longitudinal subject list
    long_df = df[df['base_subject_id'].isin(longitudinal_subjects.index)].copy()
    long_df = long_df.sort_values(['base_subject_id', 'age_months'])
    
    output_file = '/Users/stepher/Desktop/Actigraphy2/results/longitudinal_subjects.csv'
    long_df.to_csv(output_file, index=False)
    print(f"\n✓ Longitudinal data saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("LONGITUDINAL DATA SUMMARY")
    print("="*70)
    print(f"Number of longitudinal subjects: {len(longitudinal_subjects)}")
    print(f"Average timepoints per subject: {longitudinal_subjects.mean():.1f}")
    print(f"Max timepoints for any subject: {longitudinal_subjects.max()}")
    print(f"Total longitudinal observations: {long_df.shape[0]}")
    
    print("\nAge span analysis:")
    for subject in longitudinal_subjects.index:
        subj_data = df[df['base_subject_id'] == subject]
        age_range = subj_data['age_months'].max() - subj_data['age_months'].min()
        print(f"  {subject}: {age_range} months span")

else:
    print("\n" + "="*70)
    print("NO LONGITUDINAL DATA FOUND")
    print("="*70)
    print("\nAll subjects appear only once in the dataset.")
    print("This is CROSS-SECTIONAL data (different subjects at different ages).")
    print("\nFor longitudinal analysis, you would need the same subjects")
    print("measured at multiple timepoints, e.g.:")
    print("  - TOSS_102_16mos")
    print("  - TOSS_102_21mos")
    print("  - TOSS_102_26mos")

print("="*70)