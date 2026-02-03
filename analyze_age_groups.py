"""
analyze_age_groups.py
Create age group comparison visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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

# ============================================================================
# FIGURE 1: Sleep Duration Development (Line plot with error bars)
# ============================================================================

print("Creating Figure 1: Sleep Duration Development...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Total Sleep Time
ax = axes[0]
means = df.groupby('age_months')['sleep_time_mean'].mean()
stds = df.groupby('age_months')['sleep_time_mean'].std()
sems = df.groupby('age_months')['sleep_time_mean'].sem()

# Convert to hours
means_hours = means / 60
stds_hours = stds / 60
sems_hours = sems / 60

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

# Color boxes
for patch in bp['boxes']:
    patch.set_facecolor('#2E86AB')
    patch.set_alpha(0.7)

# Overlay individual points
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
# FIGURE 3: Nap Patterns by Age (Grouped Bar Chart)
# ============================================================================

print("Creating Figure 3: Nap Patterns by Age...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Sleep Period Counts
ax = axes[0]

morning_counts = df.groupby('age_months')['morning_count'].mean()
midday_counts = df.groupby('age_months')['afternoon_count'].mean()
evening_counts = df.groupby('age_months')['evening_count'].mean()

x = np.arange(len(age_groups))
width = 0.25

ax.bar(x - width, morning_counts, width, label='Morning (10:00-11:19)', color='#A23B72', alpha=0.8)
ax.bar(x, midday_counts, width, label='Midday (11:20-17:59)', color='#F18F01', alpha=0.8)
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
plt.savefig(f'{output_folder}/figure3_nap_patterns_by_age.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure3_nap_patterns_by_age.png")
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
        'Efficiency (%)': f"{age_data['efficiency_mean'].mean():.1f} ± {age_data['efficiency_mean'].std():.1f}",
        'Fragmentation': f"{age_data['fragmentation_mean'].mean():.2f} ± {age_data['fragmentation_mean'].std():.2f}",
        'Morning Naps': f"{age_data['morning_count'].mean():.1f}",
        'Midday Naps': f"{age_data['afternoon_count'].mean():.1f}",
        'Evening Sleep': f"{age_data['evening_count'].mean():.1f}"
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
print("="*70)