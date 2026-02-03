"""
attractor_analysis.py
Dynamical systems attractor analysis using return maps

Analyses:
- Return maps (bedtime, wake time, duration)
- Attractor identification
- System stability quantification
- Individual vs population dynamics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/age_groups'
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("ATTRACTOR & RETURN MAP ANALYSIS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading per-day data...")

subject_data = {}

for file_path in Path(input_folder).glob('*.csv'):
    subject_id = file_path.stem
    
    try:
        df = pd.read_csv(file_path)
        sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
        
        if len(sleep_df) == 0:
            continue
        
        # Clean data
        for col in ['sleep_time']:
            if col in sleep_df.columns:
                sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
        
        sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
        
        if len(sleep_df) == 0:
            continue
        
        # Parse times
        sleep_df['bedtime'] = pd.to_datetime(sleep_df['start_date'] + ' ' + sleep_df['start_time'])
        sleep_df['waketime'] = pd.to_datetime(sleep_df['end_date'] + ' ' + sleep_df['end_time'])
        sleep_df['bedtime_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
        sleep_df['waketime_hour'] = sleep_df['waketime'].dt.hour + sleep_df['waketime'].dt.minute / 60
        
        # Daily aggregation
        daily = sleep_df.groupby('start_date').agg({
            'sleep_time': 'sum',
            'bedtime_hour': 'mean',
            'waketime_hour': 'mean'
        }).reset_index()
        
        daily = daily.sort_values('start_date')
        
        if len(daily) >= 2:  # Need at least 2 days for return map
            subject_data[subject_id] = daily
        
    except Exception as e:
        continue

print(f"Loaded {len(subject_data)} subjects with 2+ days of data\n")

# Extract age
def extract_age(subject_id):
    try:
        parts = subject_id.split('_')
        age_part = [p for p in parts if 'mos' in p.lower()]
        if age_part:
            return int(age_part[0].replace('mos', '').replace('mo', ''))
        return None
    except:
        return None

# ============================================================================
# RETURN MAP ANALYSIS
# ============================================================================

print("Calculating return map coordinates...")

return_map_data = []

for subject_id, daily_data in subject_data.items():
    age = extract_age(subject_id)
    
    # Bedtime return map
    bedtimes = daily_data['bedtime_hour'].values
    for i in range(len(bedtimes) - 1):
        if not np.isnan(bedtimes[i]) and not np.isnan(bedtimes[i+1]):
            deviation = abs(bedtimes[i+1] - bedtimes[i])
            return_map_data.append({
                'subject_id': subject_id,
                'age_months': age,
                'variable': 'bedtime',
                'day_n': bedtimes[i],
                'day_n1': bedtimes[i+1],
                'deviation': deviation,
                'deviation_minutes': deviation * 60
            })
    
    # Duration return map
    durations = daily_data['sleep_time'].values
    for i in range(len(durations) - 1):
        if not np.isnan(durations[i]) and not np.isnan(durations[i+1]):
            deviation = abs(durations[i+1] - durations[i])
            return_map_data.append({
                'subject_id': subject_id,
                'age_months': age,
                'variable': 'duration',
                'day_n': durations[i],
                'day_n1': durations[i+1],
                'deviation': deviation,
                'deviation_minutes': deviation
            })
    
    # Wake time return map
    waketimes = daily_data['waketime_hour'].values
    for i in range(len(waketimes) - 1):
        if not np.isnan(waketimes[i]) and not np.isnan(waketimes[i+1]):
            deviation = abs(waketimes[i+1] - waketimes[i])
            return_map_data.append({
                'subject_id': subject_id,
                'age_months': age,
                'variable': 'waketime',
                'day_n': waketimes[i],
                'day_n1': waketimes[i+1],
                'deviation': deviation,
                'deviation_minutes': deviation * 60
            })

return_map_df = pd.DataFrame(return_map_data)
return_map_df = return_map_df[return_map_df['age_months'].notna()]

print(f"Calculated {len(return_map_df)} return map points\n")

# Save
return_map_df.to_csv(f'{output_folder}/return_map_coordinates.csv', index=False)
print(f"✓ Saved: return_map_coordinates.csv\n")

# ============================================================================
# STABILITY METRICS
# ============================================================================

print("Calculating stability metrics...")

stability_results = []

for subject_id in return_map_df['subject_id'].unique():
    subject_map = return_map_df[return_map_df['subject_id'] == subject_id]
    age = subject_map['age_months'].iloc[0]
    
    # Bedtime stability
    bedtime_data = subject_map[subject_map['variable'] == 'bedtime']
    if len(bedtime_data) > 0:
        bed_deviation_mean = bedtime_data['deviation_minutes'].mean()
        bed_deviation_std = bedtime_data['deviation_minutes'].std()
        
        # Calculate distance from diagonal (stability metric)
        bed_distances = np.abs(bedtime_data['day_n1'] - bedtime_data['day_n'])
        bed_stability = 1 / (1 + bed_distances.mean())  # 0-1 scale, higher = more stable
    else:
        bed_deviation_mean = None
        bed_deviation_std = None
        bed_stability = None
    
    # Duration stability
    duration_data = subject_map[subject_map['variable'] == 'duration']
    if len(duration_data) > 0:
        dur_deviation_mean = duration_data['deviation_minutes'].mean()
        dur_deviation_std = duration_data['deviation_minutes'].std()
        
        dur_distances = np.abs(duration_data['day_n1'] - duration_data['day_n'])
        dur_stability = 1 / (1 + dur_distances.mean()/60)  # Normalize by hour
    else:
        dur_deviation_mean = None
        dur_deviation_std = None
        dur_stability = None
    
    stability_results.append({
        'subject_id': subject_id,
        'age_months': age,
        'bedtime_deviation_mean_min': bed_deviation_mean,
        'bedtime_deviation_std_min': bed_deviation_std,
        'bedtime_stability': bed_stability,
        'duration_deviation_mean_min': dur_deviation_mean,
        'duration_deviation_std_min': dur_deviation_std,
        'duration_stability': dur_stability
    })

stability_df = pd.DataFrame(stability_results)

# Save
stability_df.to_csv(f'{output_folder}/attractor_stability_metrics.csv', index=False)
print(f"✓ Saved: attractor_stability_metrics.csv\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")

age_groups = sorted(return_map_df['age_months'].unique())
colors_age = {16: '#A23B72', 21: '#F18F01', 26: '#2E86AB', 31: '#C73E1D'}

# ============================================================================
# Figure 1: Return Maps
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bedtime Return Map
ax = axes[0, 0]
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'bedtime')]
    ax.scatter(age_data['day_n'], age_data['day_n1'],
              s=30, alpha=0.4, color=colors_age.get(age, 'gray'),
              label=f'{age} months')

# Diagonal line
ax.plot([18, 24], [18, 24], 'k--', linewidth=2, alpha=0.5, label='Perfect stability')
ax.set_xlabel('Bedtime Day N (hours)', fontweight='bold')
ax.set_ylabel('Bedtime Day N+1 (hours)', fontweight='bold')
ax.set_title('Bedtime Return Map', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Format axes
x_ticks = [18, 19, 20, 21, 22, 23, 24]
x_labels = ['6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM', '12 AM']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_yticks(x_ticks)
ax.set_yticklabels(x_labels)

# Duration Return Map
ax = axes[0, 1]
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'duration')]
    ax.scatter(age_data['day_n'] / 60, age_data['day_n1'] / 60,
              s=30, alpha=0.4, color=colors_age.get(age, 'gray'),
              label=f'{age} months')

# Diagonal
dur_min = return_map_df[return_map_df['variable'] == 'duration']['day_n'].min() / 60
dur_max = return_map_df[return_map_df['variable'] == 'duration']['day_n'].max() / 60
ax.plot([dur_min, dur_max], [dur_min, dur_max], 'k--', linewidth=2, alpha=0.5)

ax.set_xlabel('Duration Day N (hours)', fontweight='bold')
ax.set_ylabel('Duration Day N+1 (hours)', fontweight='bold')
ax.set_title('Duration Return Map', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Wake Time Return Map
ax = axes[1, 0]
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'waketime')]
    ax.scatter(age_data['day_n'], age_data['day_n1'],
              s=30, alpha=0.4, color=colors_age.get(age, 'gray'),
              label=f'{age} months')

ax.plot([5, 10], [5, 10], 'k--', linewidth=2, alpha=0.5)
ax.set_xlabel('Wake Time Day N (hours)', fontweight='bold')
ax.set_ylabel('Wake Time Day N+1 (hours)', fontweight='bold')
ax.set_title('Wake Time Return Map', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

y_ticks = [5, 6, 7, 8, 9, 10]
y_labels = ['5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM']
ax.set_xticks(y_ticks)
ax.set_xticklabels(y_labels, rotation=45, ha='right')
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Deviation from Diagonal
ax = axes[1, 1]

bedtime_dev = []
duration_dev = []

for age in age_groups:
    bed_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'bedtime')]
    bedtime_dev.append(bed_data['deviation_minutes'].mean())
    
    dur_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'duration')]
    duration_dev.append(dur_data['deviation_minutes'].mean())

ax.plot(age_groups, bedtime_dev, marker='o', markersize=10, linewidth=2,
       color='#2E86AB', label='Bedtime')
ax.plot(age_groups, duration_dev, marker='s', markersize=10, linewidth=2,
       color='#F18F01', label='Duration')

ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Mean Day-to-Day Change (minutes)', fontweight='bold')
ax.set_title('System Stability by Age', fontweight='bold')
ax.set_xticks(age_groups)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/attractor_fig1_return_maps.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: attractor_fig1_return_maps.png")
plt.close()

# ============================================================================
# Figure 2: Attractor Characteristics
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Stability Score by Age
ax = axes[0, 0]
stab_means = stability_df.groupby('age_months')['bedtime_stability'].mean()
stab_stds = stability_df.groupby('age_months')['bedtime_stability'].std()

ax.errorbar(age_groups, stab_means, yerr=stab_stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB')
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Bedtime Stability Score', fontweight='bold')
ax.set_title('Attractor Strength by Age', fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)

# Individual Stability Scores
ax = axes[0, 1]
for age in age_groups:
    age_data = stability_df[stability_df['age_months'] == age]
    x_pos = [age] * len(age_data)
    ax.scatter(x_pos, age_data['bedtime_stability'],
              s=80, alpha=0.5, color=colors_age.get(age, 'gray'))

ax.plot(age_groups, stab_means, 'ko-', markersize=12, linewidth=3,
       label='Mean', zorder=10)
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Bedtime Stability Score', fontweight='bold')
ax.set_title('Individual Attractor Stability', fontweight='bold')
ax.set_xticks(age_groups)
ax.legend()
ax.grid(True, alpha=0.3)

# Deviation Distribution
ax = axes[1, 0]
data_list = [return_map_df[(return_map_df['age_months'] == age) & 
                           (return_map_df['variable'] == 'bedtime')]['deviation_minutes'].dropna()
            for age in age_groups]

bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('#2E86AB')
    patch.set_alpha(0.7)

ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Day-to-Day Bedtime Change (minutes)', fontweight='bold')
ax.set_title('Variability Distribution', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Correlation: Deviation vs Mean Bedtime
ax = axes[1, 1]

subject_means = []
subject_devs = []
subject_ages = []

for subject_id in stability_df['subject_id'].unique():
    subj_return = return_map_df[(return_map_df['subject_id'] == subject_id) & 
                                (return_map_df['variable'] == 'bedtime')]
    if len(subj_return) > 0:
        subject_means.append(subj_return['day_n'].mean())
        subject_devs.append(subj_return['deviation_minutes'].mean())
        subject_ages.append(subj_return['age_months'].iloc[0])

for age in age_groups:
    mask = np.array(subject_ages) == age
    if mask.any():
        ax.scatter(np.array(subject_means)[mask], np.array(subject_devs)[mask],
                  s=80, alpha=0.6, color=colors_age.get(age, 'gray'),
                  label=f'{age} months')

ax.set_xlabel('Mean Bedtime (hours)', fontweight='bold')
ax.set_ylabel('Mean Day-to-Day Change (minutes)', fontweight='bold')
ax.set_title('Attractor Position vs Stability', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/attractor_fig2_characteristics.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: attractor_fig2_characteristics.png")
plt.close()

print("\n" + "="*70)
print("ATTRACTOR ANALYSIS COMPLETE")
print("="*70)
print(f"\nKey Findings:")
print(f"  • Mean bedtime deviation: {return_map_df[return_map_df['variable']=='bedtime']['deviation_minutes'].mean():.1f} minutes")
print(f"  • Mean duration deviation: {return_map_df[return_map_df['variable']=='duration']['deviation_minutes'].mean():.1f} minutes")
print(f"  • Mean bedtime stability: {stability_df['bedtime_stability'].mean():.3f}")
print("\nFiles saved:")
print("  • return_map_coordinates.csv")
print("  • attractor_stability_metrics.csv")
print("  • attractor_fig1_return_maps.png")
print("  • attractor_fig2_characteristics.png")
print("="*70)