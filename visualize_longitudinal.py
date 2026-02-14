"""
visualize_longitudinal.py
Visualize individual developmental trajectories for longitudinal subjects
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime, timedelta

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration - UPDATE THESE PATHS TO MATCH YOUR SYSTEM
input_csv = '/Users/stepher/Desktop/Actigraphy2/results/longitudinal_subjects.csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/longitudinal'

# Check if input file exists
if not os.path.exists(input_csv):
    print(f"\n⚠️  Data file not found: {input_csv}")
    print("\nPlease update the 'input_csv' path in the script (around line 18)")
    exit()

print(f"Using data file: {input_csv}")
print(f"Output folder: {output_folder}")

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("LONGITUDINAL TRAJECTORY ANALYSIS")
print("="*70)
print(f"Input file: {input_csv}")
print(f"Output folder: {output_folder}")
print("="*70)

# Load data
df = pd.read_csv(input_csv)

print(f"Loaded {len(df)} observations")

# Check if this is the all_subjects file (no base_subject_id) or longitudinal file
if 'base_subject_id' not in df.columns and 'subject_id' in df.columns:
    print("\nDetected all_subjects_summary.csv format")
    print("Extracting base subject IDs and filtering for longitudinal subjects...")
    
    # Extract base subject ID and age from subject_id (e.g., "TOSS_105_16mos" -> "TOSS_105", 16)
    df['base_subject_id'] = df['subject_id'].str.extract(r'([A-Z]+_\d+)_')[0]
    df['age_months'] = df['subject_id'].str.extract(r'_(\d+)mos')[0].astype(int)
    
    # Filter for only subjects with multiple timepoints
    subject_counts = df['base_subject_id'].value_counts()
    longitudinal_subject_ids = subject_counts[subject_counts > 1].index.tolist()
    
    print(f"Found {len(longitudinal_subject_ids)} subjects with multiple timepoints:")
    for subj_id in sorted(longitudinal_subject_ids):
        count = subject_counts[subj_id]
        ages = sorted(df[df['base_subject_id'] == subj_id]['age_months'].tolist())
        print(f"  {subj_id}: {count} timepoints (ages: {ages})")
    
    # Filter to only longitudinal subjects
    df = df[df['base_subject_id'].isin(longitudinal_subject_ids)].copy()
    print(f"\nFiltered to {len(df)} observations from longitudinal subjects")

print(f"Unique subjects: {df['base_subject_id'].nunique()}")
print()

# ============================================================================
# Convert time string columns to minutes from midnight
# ============================================================================

def parse_time_string_to_minutes(time_str):
    """
    Convert time string like '7:44 PM' or '10:31 AM' to minutes from midnight.
    Returns float representing minutes from midnight (0-1440).
    """
    if pd.isna(time_str) or time_str == '' or time_str == 'nan':
        return np.nan
    
    try:
        # Parse the time string
        time_obj = pd.to_datetime(time_str, format='%I:%M %p').time()
        # Convert to minutes from midnight
        minutes = time_obj.hour * 60 + time_obj.minute
        return float(minutes)
    except:
        return np.nan

# Convert all timing columns from string format to minutes
timing_columns = [
    'morning_bedtime_mean', 'morning_waketime_mean',
    'afternoon_bedtime_mean', 'afternoon_waketime_mean',
    'evening_bedtime_mean', 'evening_waketime_mean'
]

for col in timing_columns:
    if col in df.columns:
        # Check if the column is string type (not already numeric)
        # Use dtype.kind to handle both 'object' and 'string' dtypes
        if df[col].dtype.kind in ['O', 'U']:
            print(f"Converting {col} from time strings to minutes...")
            df[col] = df[col].apply(parse_time_string_to_minutes)

# Get list of longitudinal subjects
longitudinal_subjects = df['base_subject_id'].unique()

# ============================================================================
# FIGURE 1: Individual Sleep Duration Trajectories
# ============================================================================

print("Creating Figure 1: Individual Sleep Duration Trajectories...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Total Sleep Duration
ax = axes[0, 0]

# Plot population mean (cross-sectional)
pop_means = df.groupby('age_months')['sleep_time_mean'].mean() / 60
pop_ages = pop_means.index.tolist()
ax.plot(pop_ages, pop_means, 'k--', linewidth=3, alpha=0.3, 
       label='Population Mean', zorder=1)

# Plot individual trajectories
colors = plt.cm.tab20(np.linspace(0, 1, len(longitudinal_subjects)))
for i, subject in enumerate(longitudinal_subjects):
    subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
    ax.plot(subj_data['age_months'], subj_data['sleep_time_mean'] / 60,
           marker='o', markersize=8, linewidth=2, alpha=0.7,
           color=colors[i], label=subject, zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Sleep Time (hours/day)', fontsize=12, fontweight='bold')
ax.set_title('Individual Sleep Duration Trajectories', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Panel B: Evening Sleep Duration
ax = axes[0, 1]

pop_means = df.groupby('age_months')['evening_duration_mean'].mean() / 60
ax.plot(pop_ages, pop_means, 'k--', linewidth=3, alpha=0.3, 
       label='Population Mean', zorder=1)

for i, subject in enumerate(longitudinal_subjects):
    subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
    ax.plot(subj_data['age_months'], subj_data['evening_duration_mean'] / 60,
           marker='o', markersize=8, linewidth=2, alpha=0.7,
           color=colors[i], label=subject, zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evening Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_title('Evening Sleep Trajectories', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Panel C: Sleep Efficiency
ax = axes[1, 0]

pop_means = df.groupby('age_months')['efficiency_mean'].mean()
ax.plot(pop_ages, pop_means, 'k--', linewidth=3, alpha=0.3,
       label='Population Mean', zorder=1)

for i, subject in enumerate(longitudinal_subjects):
    subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
    ax.plot(subj_data['age_months'], subj_data['efficiency_mean'],
           marker='o', markersize=8, linewidth=2, alpha=0.7,
           color=colors[i], label=subject, zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Efficiency Trajectories', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Panel D: Nap Percentage
ax = axes[1, 1]

# Calculate nap percentage if not already present
if 'nap_percentage' not in df.columns:
    df['nap_duration'] = df['morning_duration_mean'].fillna(0) + df['afternoon_duration_mean'].fillna(0)
    df['nap_percentage'] = (df['nap_duration'] / df['sleep_time_mean']) * 100

pop_means = df.groupby('age_months')['nap_percentage'].mean()
ax.plot(pop_ages, pop_means, 'k--', linewidth=3, alpha=0.3,
       label='Population Mean', zorder=1)

for i, subject in enumerate(longitudinal_subjects):
    subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
    ax.plot(subj_data['age_months'], subj_data['nap_percentage'],
           marker='o', markersize=8, linewidth=2, alpha=0.7,
           color=colors[i], label=subject, zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Naps as % of Total Sleep', fontsize=12, fontweight='bold')
ax.set_title('Nap Dependency Trajectories', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/figure1_individual_trajectories.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure1_individual_trajectories.png")
plt.close()

# ============================================================================
# FIGURE 2: Individual Sleep Timing Trajectories (NEW)
# ============================================================================

print("Creating Figure 2: Individual Sleep Timing Trajectories...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def convert_minutes_to_clock_time(minutes):
    """Convert minutes from midnight to clock time (hours as decimal)"""
    # Handle values that represent times after midnight the next day
    minutes = minutes % 1440  # Wrap to 24-hour format
    hours = minutes / 60
    return hours

# Panel A: Bedtime (Evening Sleep Onset)
ax = axes[0, 0]

# Check if timing columns exist, use appropriate column names from the actual dataset
timing_cols = {
    'bedtime': ['evening_bedtime_mean', 'evening_onset_mean', 'bedtime_mean', 'sleep_onset_mean'],
    'waketime': ['evening_waketime_mean', 'evening_offset_mean', 'waketime_mean', 'sleep_offset_mean'],
    'morning_bedtime': ['morning_bedtime_mean'],
    'morning_waketime': ['morning_waketime_mean'],
    'afternoon_bedtime': ['afternoon_bedtime_mean'],
    'afternoon_waketime': ['afternoon_waketime_mean']
}

# Find which columns are available
available_cols = {}
for key, possible_cols in timing_cols.items():
    for col in possible_cols:
        if col in df.columns:
            available_cols[key] = col
            break

# Calculate sleep midpoint if we have bedtime and waketime
if 'bedtime' in available_cols and 'waketime' in available_cols:
    bedtime_col = available_cols['bedtime']
    waketime_col = available_cols['waketime']
    
    # Debug: Check if columns are numeric
    print(f"  Bedtime column ({bedtime_col}) dtype: {df[bedtime_col].dtype}")
    print(f"  Waketime column ({waketime_col}) dtype: {df[waketime_col].dtype}")
    
    # Make sure columns are numeric
    if df[bedtime_col].dtype.kind in ['O', 'U']:
        print(f"  Converting {bedtime_col} to numeric...")
        df[bedtime_col] = df[bedtime_col].apply(parse_time_string_to_minutes)
    if df[waketime_col].dtype.kind in ['O', 'U']:
        print(f"  Converting {waketime_col} to numeric...")
        df[waketime_col] = df[waketime_col].apply(parse_time_string_to_minutes)
    
    def calculate_midpoint(bedtime, waketime):
        """Calculate sleep midpoint handling day wraparound"""
        if pd.isna(bedtime) or pd.isna(waketime):
            return np.nan
        # If waketime is less than bedtime, add 24 hours (1440 minutes)
        if waketime < bedtime:
            waketime = waketime + 1440
        midpoint = (bedtime + waketime) / 2
        # Wrap back to 0-1440 range
        return midpoint % 1440
    
    df['calculated_midpoint'] = df.apply(
        lambda row: calculate_midpoint(row[bedtime_col], row[waketime_col]),
        axis=1
    )
    available_cols['midpoint'] = 'calculated_midpoint'

# Panel A: Bedtime
if 'bedtime' in available_cols:
    bedtime_col = available_cols['bedtime']
    
    # Calculate population mean
    pop_data = df.groupby('age_months')[bedtime_col].mean()
    pop_data_hours = pop_data.apply(convert_minutes_to_clock_time)
    ax.plot(pop_data.index.tolist(), pop_data_hours, 'k--', linewidth=3, alpha=0.3,
           label='Population Mean', zorder=1)
    
    # Plot individual trajectories
    for i, subject in enumerate(longitudinal_subjects):
        subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
        bedtimes_hours = subj_data[bedtime_col].apply(convert_minutes_to_clock_time)
        ax.plot(subj_data['age_months'], bedtimes_hours,
               marker='o', markersize=8, linewidth=2, alpha=0.7,
               color=colors[i], label=subject, zorder=2)
    
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bedtime (hour of day)', fontsize=12, fontweight='bold')
    ax.set_title('Bedtime Trajectories', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show clock times
    ax.set_ylim(17, 24)  # 5 PM to midnight typically
    yticks = np.arange(18, 24, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(h)}:00" for h in yticks])
else:
    ax.text(0.5, 0.5, 'Bedtime data not available', 
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_title('Bedtime Trajectories', fontsize=13, fontweight='bold')

# Panel B: Wake Time
ax = axes[0, 1]

if 'waketime' in available_cols:
    waketime_col = available_cols['waketime']
    
    # Calculate population mean
    pop_data = df.groupby('age_months')[waketime_col].mean()
    pop_data_hours = pop_data.apply(convert_minutes_to_clock_time)
    ax.plot(pop_data.index.tolist(), pop_data_hours, 'k--', linewidth=3, alpha=0.3,
           label='Population Mean', zorder=1)
    
    # Plot individual trajectories
    for i, subject in enumerate(longitudinal_subjects):
        subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
        waketimes_hours = subj_data[waketime_col].apply(convert_minutes_to_clock_time)
        ax.plot(subj_data['age_months'], waketimes_hours,
               marker='o', markersize=8, linewidth=2, alpha=0.7,
               color=colors[i], label=subject, zorder=2)
    
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Wake Time (hour of day)', fontsize=12, fontweight='bold')
    ax.set_title('Morning Wake Time Trajectories', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show clock times
    ax.set_ylim(5, 10)  # 5 AM to 10 AM typically
    yticks = np.arange(5, 11, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(h)}:00" for h in yticks])
else:
    ax.text(0.5, 0.5, 'Wake time data not available', 
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_title('Morning Wake Time Trajectories', fontsize=13, fontweight='bold')

# Panel C: Sleep Midpoint
ax = axes[1, 0]

if 'midpoint' in available_cols:
    midpoint_col = available_cols['midpoint']
    
    # Calculate population mean
    pop_data = df.groupby('age_months')[midpoint_col].mean()
    pop_data_hours = pop_data.apply(convert_minutes_to_clock_time)
    ax.plot(pop_data.index.tolist(), pop_data_hours, 'k--', linewidth=3, alpha=0.3,
           label='Population Mean', zorder=1)
    
    # Plot individual trajectories
    for i, subject in enumerate(longitudinal_subjects):
        subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
        midpoints_hours = subj_data[midpoint_col].apply(convert_minutes_to_clock_time)
        ax.plot(subj_data['age_months'], midpoints_hours,
               marker='o', markersize=8, linewidth=2, alpha=0.7,
               color=colors[i], label=subject, zorder=2)
    
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sleep Midpoint (hour)', fontsize=12, fontweight='bold')
    ax.set_title('Sleep Midpoint Trajectories', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show clock times (typically midnight to 4 AM)
    yticks = [0, 1, 2, 3, 4]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['00:00', '01:00', '02:00', '03:00', '04:00'])
else:
    ax.text(0.5, 0.5, 'Sleep midpoint data not available', 
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_title('Sleep Midpoint Trajectories', fontsize=13, fontweight='bold')

# Panel D: Time in Bed (Sleep Opportunity Window) OR Morning Nap Start Time
ax = axes[1, 1]

# Calculate time in bed if both bedtime and waketime are available
if 'bedtime' in available_cols and 'waketime' in available_cols:
    bedtime_col = available_cols['bedtime']
    waketime_col = available_cols['waketime']
    
    # Calculate time in bed for each row
    df['time_in_bed_hours'] = df.apply(
        lambda row: ((row[waketime_col] - row[bedtime_col] + 1440) % 1440) / 60 
        if pd.notna(row[bedtime_col]) and pd.notna(row[waketime_col]) else np.nan,
        axis=1
    )
    
    # Calculate population mean
    pop_data = df.groupby('age_months')['time_in_bed_hours'].mean()
    ax.plot(pop_data.index.tolist(), pop_data, 'k--', linewidth=3, alpha=0.3,
           label='Population Mean', zorder=1)
    
    # Plot individual trajectories
    for i, subject in enumerate(longitudinal_subjects):
        subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
        ax.plot(subj_data['age_months'], subj_data['time_in_bed_hours'],
               marker='o', markersize=8, linewidth=2, alpha=0.7,
               color=colors[i], label=subject, zorder=2)
    
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time in Bed (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Sleep Opportunity Window', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# Alternative: Show morning nap timing if available
elif 'morning_bedtime' in available_cols:
    morning_nap_col = available_cols['morning_bedtime']
    
    # Calculate population mean
    pop_data = df.groupby('age_months')[morning_nap_col].mean()
    pop_data_hours = pop_data.apply(convert_minutes_to_clock_time)
    ax.plot(pop_data.index.tolist(), pop_data_hours, 'k--', linewidth=3, alpha=0.3,
           label='Population Mean', zorder=1)
    
    # Plot individual trajectories
    for i, subject in enumerate(longitudinal_subjects):
        subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
        if morning_nap_col in subj_data.columns:
            nap_times_hours = subj_data[morning_nap_col].apply(convert_minutes_to_clock_time)
            ax.plot(subj_data['age_months'], nap_times_hours,
                   marker='o', markersize=8, linewidth=2, alpha=0.7,
                   color=colors[i], label=subject, zorder=2)
    
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Morning Nap Start Time', fontsize=12, fontweight='bold')
    ax.set_title('Morning Nap Timing', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show clock times
    yticks = np.arange(8, 13, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(h)}:00" for h in yticks])

else:
    ax.text(0.5, 0.5, 'Time in bed data not available\n(requires bedtime and waketime)', 
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
    ax.set_title('Sleep Opportunity Window', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_folder}/figure2_sleep_timing_trajectories.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure2_sleep_timing_trajectories.png")
plt.close()

# ============================================================================
# FIGURE 3: Within-Subject vs Between-Subject Variability (formerly Figure 2)
# ============================================================================

print("Creating Figure 3: Within vs Between Subject Variability...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calculate within-subject and between-subject variability
metrics = {
    'Total Sleep (h)': 'sleep_time_mean',
    'Efficiency (%)': 'efficiency_mean',
    'Fragmentation': 'fragmentation_mean',
    'WASO (min)': 'wake_time_mean'
}

within_subj_var = []
between_subj_var = []
metric_names = []

for name, col in metrics.items():
    # Within-subject variability (average SD within each subject)
    within_sd = []
    for subject in longitudinal_subjects:
        subj_data = df[df['base_subject_id'] == subject]
        if len(subj_data) > 1:
            within_sd.append(subj_data[col].std())
    
    within_subj_var.append(np.mean(within_sd))
    
    # Between-subject variability (SD of means across subjects)
    subject_means = []
    for subject in longitudinal_subjects:
        subj_data = df[df['base_subject_id'] == subject]
        subject_means.append(subj_data[col].mean())
    
    between_subj_var.append(np.std(subject_means))
    metric_names.append(name)

# Panel A: Comparison
ax = axes[0]
x = np.arange(len(metric_names))
width = 0.35

ax.bar(x - width/2, within_subj_var, width, label='Within-Subject', 
      color='#2E86AB', alpha=0.8)
ax.bar(x + width/2, between_subj_var, width, label='Between-Subject',
      color='#F18F01', alpha=0.8)

ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
ax.set_title('Within-Subject vs Between-Subject Variability', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Ratio (ICC-like measure)
ax = axes[1]

# Calculate ratio: between / (between + within)
# High ratio = more between-subject variation (stable within-subject)
# Low ratio = more within-subject variation (developmental change)
ratios = []
for i in range(len(within_subj_var)):
    total_var = between_subj_var[i] + within_subj_var[i]
    ratio = between_subj_var[i] / total_var if total_var > 0 else 0
    ratios.append(ratio)

ax.bar(x, ratios, color='#A23B72', alpha=0.8)
ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, 
          label='Equal variance')

ax.set_ylabel('Between / (Between + Within)', fontsize=12, fontweight='bold')
ax.set_title('Variance Decomposition', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add interpretation text
ax.text(0.5, 0.95, 'High = Stable trait\nLow = Developmental change',
       transform=ax.transAxes, fontsize=9, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_folder}/figure3_variance_decomposition.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure3_variance_decomposition.png")
plt.close()

# ============================================================================
# FIGURE 4: Individual Subject Cards (formerly Figure 3)
# ============================================================================

print("Creating Figure 4: Individual Subject Profile Cards...")

# Create one detailed figure per subject
for subject in longitudinal_subjects:
    subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
    
    if len(subj_data) < 2:
        continue  # Skip single timepoint subjects
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Developmental Profile: {subject}', fontsize=16, fontweight='bold')
    
    ages = subj_data['age_months'].tolist()
    
    # Plot 1: Sleep Duration
    ax = axes[0, 0]
    ax.plot(ages, subj_data['sleep_time_mean'] / 60, 'o-', linewidth=2, 
           markersize=10, color='#2E86AB')
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel('Hours', fontweight='bold')
    ax.set_title('Total Sleep Duration')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency
    ax = axes[0, 1]
    ax.plot(ages, subj_data['efficiency_mean'], 'o-', linewidth=2,
           markersize=10, color='#F18F01')
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel('Percentage', fontweight='bold')
    ax.set_title('Sleep Efficiency')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fragmentation
    ax = axes[0, 2]
    ax.plot(ages, subj_data['fragmentation_mean'], 'o-', linewidth=2,
           markersize=10, color='#A23B72')
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel('Index', fontweight='bold')
    ax.set_title('Sleep Fragmentation')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Nap Percentage
    ax = axes[1, 0]
    nap_pct = subj_data['nap_percentage']
    ax.plot(ages, nap_pct, 'o-', linewidth=2, markersize=10, color='#C73E1D')
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel('Percentage', fontweight='bold')
    ax.set_title('Naps as % of Total Sleep')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Sleep Consolidation
    ax = axes[1, 1]
    ax.plot(ages, subj_data['avg_intervals_per_day'], 'o-', linewidth=2,
           markersize=10, color='#2E86AB')
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Sleep Intervals Per Day')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary Table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary text
    summary_text = f"Subject: {subject}\n\n"
    summary_text += f"Timepoints: {len(subj_data)}\n"
    summary_text += f"Age range: {ages[0]}-{ages[-1]} months\n"
    summary_text += f"Age span: {ages[-1] - ages[0]} months\n\n"
    
    # Calculate changes
    sleep_change = (subj_data['sleep_time_mean'].iloc[-1] - subj_data['sleep_time_mean'].iloc[0]) / 60
    eff_change = subj_data['efficiency_mean'].iloc[-1] - subj_data['efficiency_mean'].iloc[0]
    nap_change = nap_pct.iloc[-1] - nap_pct.iloc[0]
    
    summary_text += "Changes over time:\n"
    summary_text += f"  Sleep: {sleep_change:+.1f} hours\n"
    summary_text += f"  Efficiency: {eff_change:+.1f}%\n"
    summary_text += f"  Nap %: {nap_change:+.1f}%\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/figure4_profile_{subject}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: figure4_profile_{subject}.png")
    plt.close()

# ============================================================================
# Statistical Summary
# ============================================================================

print("\nCreating statistical summary...")

summary_stats = []

for subject in longitudinal_subjects:
    subj_data = df[df['base_subject_id'] == subject].sort_values('age_months')
    
    if len(subj_data) < 2:
        continue
    
    ages = subj_data['age_months'].tolist()
    
    # Calculate rate of change (per month)
    age_span = ages[-1] - ages[0]
    
    sleep_rate = (subj_data['sleep_time_mean'].iloc[-1] - subj_data['sleep_time_mean'].iloc[0]) / age_span / 60
    eff_rate = (subj_data['efficiency_mean'].iloc[-1] - subj_data['efficiency_mean'].iloc[0]) / age_span
    nap_rate = (subj_data['nap_percentage'].iloc[-1] - subj_data['nap_percentage'].iloc[0]) / age_span
    
    stats = {
        'Subject': subject,
        'Timepoints': len(subj_data),
        'Age Range': f"{ages[0]}-{ages[-1]}",
        'Age Span (months)': age_span,
        'Sleep Change (h/month)': f"{sleep_rate:.2f}",
        'Efficiency Change (%/month)': f"{eff_rate:.2f}",
        'Nap % Change (%/month)': f"{nap_rate:.2f}"
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f'{output_folder}/longitudinal_summary_statistics.csv', index=False)
print(f"  ✓ Saved: longitudinal_summary_statistics.csv")

print("\n" + "="*70)
print("LONGITUDINAL ANALYSIS SUMMARY")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"All longitudinal visualizations saved to: {output_folder}")
print("\nGenerated figures:")
print("  - Figure 1: Individual Sleep Duration Trajectories")
print("  - Figure 2: Individual Sleep Timing Trajectories (NEW)")
print("  - Figure 3: Within vs Between Subject Variability")
print("  - Figure 4: Individual Subject Profile Cards")
print("="*70)