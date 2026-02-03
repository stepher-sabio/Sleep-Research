"""
visualize_longitudinal.py
Visualize individual developmental trajectories for longitudinal subjects
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_csv = '/Users/stepher/Desktop/Actigraphy2/results/longitudinal_subjects.csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/longitudinal'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("LONGITUDINAL TRAJECTORY ANALYSIS")
print("="*70)

# Check if longitudinal data exists
if not os.path.exists(input_csv):
    print("\n⚠ No longitudinal data file found!")
    print("Run 'detect_longitudinal.py' first to check for longitudinal subjects.")
    print("\nIf you have no longitudinal data, this analysis cannot be performed.")
    exit()

# Load data
df = pd.read_csv(input_csv)

print(f"Loaded {len(df)} observations")
print(f"Unique subjects: {df['base_subject_id'].nunique()}")
print()

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
# FIGURE 2: Within-Subject vs Between-Subject Variability
# ============================================================================

print("Creating Figure 2: Within vs Between Subject Variability...")

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
plt.savefig(f'{output_folder}/figure2_variance_decomposition.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: figure2_variance_decomposition.png")
plt.close()

# ============================================================================
# FIGURE 3: Individual Subject Cards (Detailed Profiles)
# ============================================================================

print("Creating Figure 3: Individual Subject Profile Cards...")

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
    plt.savefig(f'{output_folder}/profile_{subject}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: profile_{subject}.png")
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
print("="*70)