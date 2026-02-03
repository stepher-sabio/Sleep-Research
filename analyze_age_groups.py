"""
analyze_age_groups.py
Create comprehensive age group comparison visualizations
Includes: timings, durations, naps vs nighttime, quality metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_csv = '/Users/stepher/Desktop/Actigraphy2/results/all_subjects_summary.csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/age_groups'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("AGE GROUP ANALYSIS")
print("="*70)
print(f"Input: {input_csv}")
print(f"Output: {output_folder}")
print("="*70 + "\n")

# Load data
df = pd.read_csv(input_csv)

# Extract age from subject_id
def extract_age(subject_id):
    """Extract age in months from subject ID (e.g., 'TOSS_102_16mos' → 16)"""
    try:
        parts = subject_id.split('_')
        age_part = [p for p in parts if 'mos' in p.lower()]
        if age_part:
            age_str = age_part[0].replace('mos', '').replace('mo', '')
            return int(age_str)
        return None
    except:
        return None

df['age_months'] = df['subject_id'].apply(extract_age)

# Remove subjects without valid age
df = df[df['age_months'].notna()]
df['age_months'] = df['age_months'].astype(int)

print(f"Total subjects: {len(df)}")
print(f"Age groups found: {sorted(df['age_months'].unique())}")
print(f"\nSubjects per age group:")
print(df['age_months'].value_counts().sort_index())
print()

# Get age groups
age_groups = sorted(df['age_months'].unique())

# Helper function to convert time strings to decimal hours
def time_to_decimal(time_str):
    """Convert time string like '7:23 PM' to decimal hours (19.38)"""
    if pd.isna(time_str) or time_str is None:
        return np.nan
    try:
        # Parse time string
        time_obj = datetime.strptime(str(time_str), '%I:%M %p')
        hours = time_obj.hour + time_obj.minute / 60
        return hours
    except:
        return np.nan

# ============================================================================
# FIGURE 1: Sleep Duration Development (Line plot with error bars)
# ============================================================================

print("Creating Figure 1: Sleep Duration Development...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Total Sleep Time
ax = axes[0]
means = df.groupby('age_months')['sleep_time_mean'].mean()
stds = df.groupby('age_months')['sleep_time_mean'].std()

means_hours = means / 60
stds_hours = stds / 60

# Plot individual points
for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['sleep_time_mean'] / 60,
              alpha=0.4, s=50, color='gray', zorder=1)

# Plot line with error bars
ax.errorbar(age_groups, means_hours, yerr=stds_hours, 
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB', label='Mean ± SD', zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Sleep Time (hours/day)', fontsize=12, fontweight='bold')
ax.set_title('Total Sleep Duration by Age', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)

# Panel B: Evening Sleep Duration
ax = axes[1]
means = df.groupby('age_months')['evening_duration_mean'].mean()
stds = df.groupby('age_months')['evening_duration_mean'].std()

means_hours = means / 60
stds_hours = stds / 60

# Plot individual points
for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['evening_duration_mean'] / 60,
              alpha=0.4, s=50, color='gray', zorder=1)

# Plot line
ax.errorbar(age_groups, means_hours, yerr=stds_hours,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB', label='Mean ± SD', zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evening Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_title('Evening Sleep Duration by Age', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)

# Panel C: Total Sleep by Period (Stacked Bar)
ax = axes[2]

# Calculate means for each period
morning_means = df.groupby('age_months')['morning_duration_mean'].mean().fillna(0) / 60
midday_means = df.groupby('age_months')['afternoon_duration_mean'].mean().fillna(0) / 60
evening_means = df.groupby('age_months')['evening_duration_mean'].mean().fillna(0) / 60

width = 0.6
x = np.arange(len(age_groups))

# Stacked bars
ax.bar(x, evening_means, width, label='Evening', color='#2E86AB')
ax.bar(x, midday_means, width, bottom=evening_means, label='Midday', color='#F18F01')
ax.bar(x, morning_means, width, bottom=evening_means + midday_means, label='Morning', color='#A23B72')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Distribution by Time of Day', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_folder}/figure1_sleep_duration_development.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure1_sleep_duration_development.png")
plt.close()

# ============================================================================
# FIGURE 2: Sleep Quality Box Plots
# ============================================================================

print("Creating Figure 2: Sleep Quality Box Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Sleep Efficiency
ax = axes[0, 0]
data_list = [df[df['age_months'] == age]['efficiency_mean'].dropna() for age in age_groups]
bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#2E86AB')
    patch.set_alpha(0.7)

for i, age in enumerate(age_groups):
    age_data = df[df['age_months'] == age]['efficiency_mean'].dropna()
    x = np.random.normal(i + 1, 0.04, size=len(age_data))
    ax.scatter(x, age_data, alpha=0.4, s=30, color='darkblue')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Efficiency by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Fragmentation
ax = axes[0, 1]
data_list = [df[df['age_months'] == age]['fragmentation_mean'].dropna() for age in age_groups]
bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#F18F01')
    patch.set_alpha(0.7)

for i, age in enumerate(age_groups):
    age_data = df[df['age_months'] == age]['fragmentation_mean'].dropna()
    x = np.random.normal(i + 1, 0.04, size=len(age_data))
    ax.scatter(x, age_data, alpha=0.4, s=30, color='darkorange')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fragmentation Index', fontsize=12, fontweight='bold')
ax.set_title('Sleep Fragmentation by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Onset Latency
ax = axes[1, 0]
data_list = [df[df['age_months'] == age]['onset_latency_mean'].dropna() for age in age_groups]
bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#A23B72')
    patch.set_alpha(0.7)

for i, age in enumerate(age_groups):
    age_data = df[df['age_months'] == age]['onset_latency_mean'].dropna()
    x = np.random.normal(i + 1, 0.04, size=len(age_data))
    ax.scatter(x, age_data, alpha=0.4, s=30, color='purple')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Onset Latency (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Onset Latency by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel D: WASO (Wake After Sleep Onset)
ax = axes[1, 1]
data_list = [df[df['age_months'] == age]['wake_time_mean'].dropna() for age in age_groups]
bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#C73E1D')
    patch.set_alpha(0.7)

for i, age in enumerate(age_groups):
    age_data = df[df['age_months'] == age]['wake_time_mean'].dropna()
    x = np.random.normal(i + 1, 0.04, size=len(age_data))
    ax.scatter(x, age_data, alpha=0.4, s=30, color='darkred')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('WASO (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Wake After Sleep Onset by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_folder}/figure2_sleep_quality_boxplots.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure2_sleep_quality_boxplots.png")
plt.close()

# ============================================================================
# FIGURE 3: Sleep Timing Trends (NEW - Bedtime/Wake Time)
# ============================================================================

print("Creating Figure 3: Sleep Timing Trends...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Convert time strings to decimal hours
df['evening_bedtime_decimal'] = df['evening_bedtime_mean'].apply(time_to_decimal)
df['evening_waketime_decimal'] = df['evening_waketime_mean'].apply(time_to_decimal)

# Panel A: Evening Bedtime Trend
ax = axes[0, 0]
means = df.groupby('age_months')['evening_bedtime_decimal'].mean()
stds = df.groupby('age_months')['evening_bedtime_decimal'].std()

# Plot individual points
for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['evening_bedtime_decimal'],
              alpha=0.4, s=50, color='gray', zorder=1)

# Plot line
ax.errorbar(age_groups, means, yerr=stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB', label='Mean ± SD', zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Bedtime (24-hour)', fontsize=12, fontweight='bold')
ax.set_title('Evening Bedtime by Age', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)

# Y-axis formatting
y_ticks = [18, 19, 20, 21, 22, 23, 24]
y_labels = ['6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM', '12 AM']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Panel B: Evening Wake Time Trend
ax = axes[0, 1]
means = df.groupby('age_months')['evening_waketime_decimal'].mean()
stds = df.groupby('age_months')['evening_waketime_decimal'].std()

# Plot individual points
for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['evening_waketime_decimal'],
              alpha=0.4, s=50, color='gray', zorder=1)

# Plot line
ax.errorbar(age_groups, means, yerr=stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#F18F01', label='Mean ± SD', zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wake Time (24-hour)', fontsize=12, fontweight='bold')
ax.set_title('Evening Wake Time by Age', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)

# Y-axis formatting
y_ticks = [5, 6, 7, 8, 9, 10]
y_labels = ['5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Panel C: Nap vs Nighttime Duration Comparison
ax = axes[1, 0]

# Calculate nap duration (morning + midday/afternoon)
df['nap_duration'] = df['morning_duration_mean'].fillna(0) + df['afternoon_duration_mean'].fillna(0)
df['nighttime_duration'] = df['evening_duration_mean']

# Prepare data for grouped bars
nap_means = []
nap_stds = []
night_means = []
night_stds = []

for age in age_groups:
    age_data = df[df['age_months'] == age]
    nap_means.append(age_data['nap_duration'].mean() / 60)
    nap_stds.append(age_data['nap_duration'].std() / 60)
    night_means.append(age_data['nighttime_duration'].mean() / 60)
    night_stds.append(age_data['nighttime_duration'].std() / 60)

x = np.arange(len(age_groups))
width = 0.35

ax.bar(x - width/2, night_means, width, yerr=night_stds, 
      label='Evening/Nighttime', color='#2E86AB', alpha=0.8, capsize=5)
ax.bar(x + width/2, nap_means, width, yerr=nap_stds,
      label='Naps (Morning + Midday)', color='#F18F01', alpha=0.8, capsize=5)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_title('Nighttime vs Nap Duration by Age', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Percentage of Sleep in Naps
ax = axes[1, 1]

# Calculate percentage
df['nap_percentage'] = (df['nap_duration'] / df['sleep_time_mean']) * 100

perc_means = []
perc_stds = []

for age in age_groups:
    age_data = df[df['age_months'] == age]
    perc_means.append(age_data['nap_percentage'].mean())
    perc_stds.append(age_data['nap_percentage'].std())

# Plot individual points
for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['nap_percentage'],
              alpha=0.4, s=50, color='gray', zorder=1)

# Plot line
ax.errorbar(age_groups, perc_means, yerr=perc_stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#A23B72', label='Mean ± SD', zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Naps as % of Total Sleep', fontsize=12, fontweight='bold')
ax.set_title('Sleep Consolidation: Nap Dependency by Age', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)

plt.tight_layout()
plt.savefig(f'{output_folder}/figure3_sleep_timing_trends.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure3_sleep_timing_trends.png")
plt.close()

# ============================================================================
# FIGURE 4: Nap Patterns by Age (Grouped Bar Chart)
# ============================================================================

print("Creating Figure 4: Nap Patterns by Age...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Sleep Period Counts
ax = axes[0]

morning_counts = df.groupby('age_months')['morning_count'].mean()
midday_counts = df.groupby('age_months')['afternoon_count'].mean()
evening_counts = df.groupby('age_months')['evening_count'].mean()

x = np.arange(len(age_groups))
width = 0.25

ax.bar(x - width, morning_counts, width, label='Morning (10:00-11:59)', color='#A23B72', alpha=0.8)
ax.bar(x, midday_counts, width, label='Midday (12:00-17:59)', color='#F18F01', alpha=0.8)
ax.bar(x + width, evening_counts, width, label='Evening (18:00-09:59)', color='#2E86AB', alpha=0.8)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Number of Sleep Periods', fontsize=12, fontweight='bold')
ax.set_title('Sleep Period Count by Time of Day', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Average Intervals Per Day
ax = axes[1]

intervals_mean = df.groupby('age_months')['avg_intervals_per_day'].mean()
intervals_std = df.groupby('age_months')['avg_intervals_per_day'].std()

# Plot individual points
for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['avg_intervals_per_day'],
              alpha=0.4, s=50, color='gray', zorder=1)

# Plot line
ax.errorbar(age_groups, intervals_mean, yerr=intervals_std,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB', label='Mean ± SD', zorder=2)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Sleep Intervals Per Day', fontsize=12, fontweight='bold')
ax.set_title('Sleep Consolidation by Age', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)

plt.tight_layout()
plt.savefig(f'{output_folder}/figure4_nap_patterns_by_age.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure4_nap_patterns_by_age.png")
plt.close()

# ============================================================================
# Summary Statistics Table
# ============================================================================

print("\nCreating summary statistics table...")

summary_stats = []

for age in age_groups:
    age_data = df[df['age_months'] == age]
    
    stats = {
        'Age (months)': age,
        'N': len(age_data),
        'Total Sleep (h)': f"{age_data['sleep_time_mean'].mean() / 60:.1f} ± {age_data['sleep_time_mean'].std() / 60:.1f}",
        'Nighttime (h)': f"{age_data['evening_duration_mean'].mean() / 60:.1f} ± {age_data['evening_duration_mean'].std() / 60:.1f}",
        'Naps (h)': f"{age_data['nap_duration'].mean() / 60:.1f} ± {age_data['nap_duration'].std() / 60:.1f}",
        'Nap %': f"{age_data['nap_percentage'].mean():.1f}%",
        'Bedtime': age_data['evening_bedtime_mean'].mode()[0] if len(age_data['evening_bedtime_mean'].mode()) > 0 else 'N/A',
        'Wake Time': age_data['evening_waketime_mean'].mode()[0] if len(age_data['evening_waketime_mean'].mode()) > 0 else 'N/A',
        'Efficiency (%)': f"{age_data['efficiency_mean'].mean():.1f} ± {age_data['efficiency_mean'].std():.1f}",
        'Fragmentation': f"{age_data['fragmentation_mean'].mean():.2f} ± {age_data['fragmentation_mean'].std():.2f}",
        'Morning Naps': f"{age_data['morning_count'].mean():.1f}",
        'Midday Naps': f"{age_data['afternoon_count'].mean():.1f}",
        'Sleep Periods/Day': f"{age_data['avg_intervals_per_day'].mean():.1f}"
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f'{output_folder}/age_group_summary_statistics.csv', index=False)
print(f"  ✓ Saved: age_group_summary_statistics.csv")

print("\n" + "="*70)
print("SUMMARY STATISTICS BY AGE GROUP")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"All visualizations saved to: {output_folder}")
print("\nGenerated figures:")
print("  - Figure 1: Sleep Duration Development (total, evening, stacked)")
print("  - Figure 2: Sleep Quality Box Plots (efficiency, fragmentation, latency, WASO)")
print("  - Figure 3: Sleep Timing Trends (bedtime, wake time, nap vs nighttime, consolidation)")
print("  - Figure 4: Nap Patterns (period counts, intervals per day)")
print("  - Summary Statistics CSV")
print("="*70)

# ============================================================================
# FIGURE 5: Correlation Heatmap (Variable Relationships)
# ============================================================================

print("Creating Figure 5: Correlation Heatmap...")

# Select key variables for correlation
corr_vars = [
    'sleep_time_mean', 'efficiency_mean', 'fragmentation_mean', 
    'onset_latency_mean', 'wake_time_mean', 'avg_intervals_per_day',
    'evening_duration_mean', 'morning_duration_mean', 'afternoon_duration_mean'
]

# Add age as a variable
df_corr = df[['age_months'] + corr_vars].copy()

# Rename columns for readability
df_corr.columns = [
    'Age', 'Total Sleep', 'Efficiency', 'Fragmentation', 
    'Onset Latency', 'WASO', 'Sleep Intervals/Day',
    'Evening Duration', 'Morning Duration', 'Midday Duration'
]

# Calculate correlation matrix
corr_matrix = df_corr.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Plot heatmap
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

# Add correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', color='black', fontsize=9)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')

ax.set_title('Sleep Variable Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_folder}/figure5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure5_correlation_heatmap.png")
plt.close()

# ============================================================================
# FIGURE 6: Phase Plot (Bedtime vs Wake Time)
# ============================================================================

print("Creating Figure 6: Phase Plot (Bedtime vs Wake Time)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: All ages combined
ax = axes[0]

# Plot each age group with different color
colors_age = {16: '#A23B72', 21: '#F18F01', 26: '#2E86AB', 31: '#C73E1D'}

for age in age_groups:
    age_data = df[df['age_months'] == age]
    
    ax.scatter(age_data['evening_bedtime_decimal'], 
              age_data['evening_waketime_decimal'],
              s=100, alpha=0.6, 
              color=colors_age.get(age, 'gray'),
              label=f'{age} months',
              edgecolors='white', linewidth=1)

ax.set_xlabel('Bedtime (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wake Time (hours)', fontsize=12, fontweight='bold')
ax.set_title('Phase Plot: Bedtime vs Wake Time (All Ages)', fontsize=13, fontweight='bold')

# Format axes
x_ticks = [18, 19, 20, 21, 22, 23, 24]
x_labels = ['6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM', '12 AM']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45)

y_ticks = [5, 6, 7, 8, 9, 10]
y_labels = ['5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

ax.legend()
ax.grid(True, alpha=0.3)

# Add diagonal reference line (equal bedtime and wake time would be on this line)
ax.plot([18, 24], [5, 11], 'k--', alpha=0.2, linewidth=1, label='12h sleep')

# Panel B: Mean trajectories with error ellipses
ax = axes[1]

for age in age_groups:
    age_data = df[df['age_months'] == age]
    
    mean_bed = age_data['evening_bedtime_decimal'].mean()
    mean_wake = age_data['evening_waketime_decimal'].mean()
    std_bed = age_data['evening_bedtime_decimal'].std()
    std_wake = age_data['evening_waketime_decimal'].std()
    
    # Plot mean point
    ax.scatter(mean_bed, mean_wake, s=200, 
              color=colors_age.get(age, 'gray'),
              marker='o', edgecolors='black', linewidth=2,
              label=f'{age} months', zorder=3)
    
    # Plot error ellipse (1 SD)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((mean_bed, mean_wake), 
                     width=2*std_bed, height=2*std_wake,
                     facecolor=colors_age.get(age, 'gray'),
                     alpha=0.2, edgecolor=colors_age.get(age, 'gray'),
                     linewidth=2)
    ax.add_patch(ellipse)

# Connect means to show trajectory
means_bed = [df[df['age_months'] == age]['evening_bedtime_decimal'].mean() for age in age_groups]
means_wake = [df[df['age_months'] == age]['evening_waketime_decimal'].mean() for age in age_groups]
ax.plot(means_bed, means_wake, 'k--', alpha=0.5, linewidth=2, zorder=1)

ax.set_xlabel('Bedtime (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wake Time (hours)', fontsize=12, fontweight='bold')
ax.set_title('Phase Plot: Developmental Trajectory', fontsize=13, fontweight='bold')

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/figure6_phase_plot_bedtime_waketime.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure6_phase_plot_bedtime_waketime.png")
plt.close()

# ============================================================================
# FIGURE 7: Variability/Consistency Analysis (Coefficient of Variation)
# ============================================================================

print("Creating Figure 7: Variability Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Function to calculate CV (coefficient of variation)
def calculate_cv(data):
    """Calculate coefficient of variation (std/mean * 100)"""
    return (data.std() / data.mean()) * 100 if data.mean() != 0 else np.nan

# Panel A: Sleep Duration Variability
ax = axes[0, 0]

duration_cv = []
for age in age_groups:
    age_data = df[df['age_months'] == age]
    cv = calculate_cv(age_data['sleep_time_mean'])
    duration_cv.append(cv)

ax.plot(age_groups, duration_cv, marker='o', markersize=10, linewidth=2,
       color='#2E86AB', label='Sleep Duration CV')
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Duration Variability by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)
ax.legend()

# Panel B: Timing Variability (Bedtime consistency)
ax = axes[0, 1]

# Calculate within-subject bedtime variability for each age group
bedtime_variability = []
for age in age_groups:
    age_data = df[df['age_months'] == age]
    # Average of individual std deviations (minutes)
    avg_std = age_data['evening_bedtime_std_minutes'].mean()
    bedtime_variability.append(avg_std)

ax.plot(age_groups, bedtime_variability, marker='o', markersize=10, linewidth=2,
       color='#F18F01', label='Bedtime Variability')
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Std Dev (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Bedtime Consistency by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)
ax.legend()

# Panel C: Efficiency Variability
ax = axes[1, 0]

efficiency_cv = []
for age in age_groups:
    age_data = df[df['age_months'] == age]
    cv = calculate_cv(age_data['efficiency_mean'])
    efficiency_cv.append(cv)

ax.plot(age_groups, efficiency_cv, marker='o', markersize=10, linewidth=2,
       color='#A23B72', label='Efficiency CV')
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Efficiency Variability by Age', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)
ax.legend()

# Panel D: Multi-metric Variability Comparison
ax = axes[1, 1]

ax.plot(age_groups, duration_cv, marker='o', markersize=8, linewidth=2,
       color='#2E86AB', label='Duration CV', alpha=0.7)
ax.plot(age_groups, efficiency_cv, marker='s', markersize=8, linewidth=2,
       color='#A23B72', label='Efficiency CV', alpha=0.7)

# Normalize bedtime variability to percentage scale
bedtime_cv_normalized = [(x / 60) * 100 for x in bedtime_variability]  # Convert to hours then %
ax.plot(age_groups, bedtime_cv_normalized, marker='^', markersize=8, linewidth=2,
       color='#F18F01', label='Bedtime Variability (normalized)', alpha=0.7)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variability', fontsize=12, fontweight='bold')
ax.set_title('System Stability: Multi-Metric Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(age_groups)
ax.legend()

plt.tight_layout()
plt.savefig(f'{output_folder}/figure7_variability_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure7_variability_analysis.png")
plt.close()

# ============================================================================
# FIGURE 8: Sleep Quality vs Duration (Two-Process Model)
# ============================================================================

print("Creating Figure 8: Quality vs Duration Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Efficiency vs Duration (all ages)
ax = axes[0, 0]

for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter(age_data['sleep_time_mean'] / 60, 
              age_data['efficiency_mean'],
              s=100, alpha=0.6,
              color=colors_age.get(age, 'gray'),
              label=f'{age} months')

ax.set_xlabel('Total Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Efficiency vs Duration (All Ages)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Efficiency vs Duration by Age (separate trends)
ax = axes[0, 1]

for age in age_groups:
    age_data = df[df['age_months'] == age]
    
    # Scatter
    ax.scatter(age_data['sleep_time_mean'] / 60, 
              age_data['efficiency_mean'],
              s=100, alpha=0.6,
              color=colors_age.get(age, 'gray'),
              label=f'{age} months')
    
    # Trend line
    if len(age_data) > 2:
        z = np.polyfit(age_data['sleep_time_mean'] / 60, 
                      age_data['efficiency_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(age_data['sleep_time_mean'].min() / 60,
                            age_data['sleep_time_mean'].max() / 60, 100)
        ax.plot(x_line, p(x_line), '--', 
               color=colors_age.get(age, 'gray'), alpha=0.5, linewidth=2)

ax.set_xlabel('Total Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('Efficiency vs Duration: Age-Specific Trends', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: Fragmentation vs Duration
ax = axes[1, 0]

for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter(age_data['sleep_time_mean'] / 60, 
              age_data['fragmentation_mean'],
              s=100, alpha=0.6,
              color=colors_age.get(age, 'gray'),
              label=f'{age} months')

ax.set_xlabel('Total Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fragmentation Index', fontsize=12, fontweight='bold')
ax.set_title('Sleep Fragmentation vs Duration', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: WASO vs Duration
ax = axes[1, 1]

for age in age_groups:
    age_data = df[df['age_months'] == age]
    ax.scatter(age_data['sleep_time_mean'] / 60, 
              age_data['wake_time_mean'],
              s=100, alpha=0.6,
              color=colors_age.get(age, 'gray'),
              label=f'{age} months')

ax.set_xlabel('Total Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('WASO (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Wake After Sleep Onset vs Duration', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/figure8_quality_vs_duration.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure8_quality_vs_duration.png")
plt.close()

# ============================================================================
# Print interpretation guide
# ============================================================================

print("\n" + "="*70)
print("INTERPRETATION GUIDE FOR DIFFERENTIAL EQUATIONS MODELING")
print("="*70)

print("\nFigure 5 (Correlation Heatmap):")
print("  - Strong correlations (>0.7 or <-0.7) suggest coupled variables")
print("  - Use correlated variables in the same differential equation")
print("  - Negative correlations suggest antagonistic relationships")

print("\nFigure 6 (Phase Plot):")
print("  - Tight clusters = stable attractors (equilibrium points)")
print("  - Spread = variability in stable states")
print("  - Ellipse size = variance in sleep timing")
print("  - Trajectory shows developmental phase shift")

print("\nFigure 7 (Variability Analysis):")
print("  - Decreasing CV with age = increasing stability")
print("  - High CV = weak homeostatic control")
print("  - Bedtime variability = circadian rhythm strength")

print("\nFigure 8 (Quality vs Duration):")
print("  - Tests two-process model predictions")
print("  - Positive slope = more sleep → better quality")
print("  - Flat/negative = saturation or optimal duration exists")

print("="*70)