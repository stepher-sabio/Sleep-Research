"""
advanced_dynamics_analysis.py
Advanced dynamical systems analysis using per-day sleep data

Analyses:
1. Regularity Metrics (IS/IV) - Circadian rhythm strength
2. Return Map Analysis - Attractor dynamics
3. Autocorrelation - Memory effects
4. Within-Subject Variability - Individual stability
5. Sleep Phenotype Clustering - Individual differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/age_groups/More Analysis'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("ADVANCED DYNAMICAL SYSTEMS ANALYSIS")
print("="*70)
print(f"Input: {input_folder}")
print(f"Output: {output_folder}")
print("="*70 + "\n")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("Loading per-day data...")

all_data = []
subject_info = {}

for file_path in Path(input_folder).glob('*.csv'):
    subject_id = file_path.stem
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Filter to SLEEP intervals
        sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
        
        if len(sleep_df) == 0:
            continue
        
        # Clean numeric columns
        numeric_cols = ['sleep_time', 'efficiency', 'onset_latency', 
                       'fragmentation', 'wake_time']
        for col in numeric_cols:
            if col in sleep_df.columns:
                sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
        
        # Remove invalid data
        sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
        
        if len(sleep_df) == 0:
            continue
        
        # Parse dates and times
        sleep_df['bedtime'] = pd.to_datetime(
            sleep_df['start_date'] + ' ' + sleep_df['start_time']
        )
        sleep_df['waketime'] = pd.to_datetime(
            sleep_df['end_date'] + ' ' + sleep_df['end_time']
        )
        
        # Extract bedtime hour (for regularity analysis)
        sleep_df['bedtime_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
        sleep_df['waketime_hour'] = sleep_df['waketime'].dt.hour + sleep_df['waketime'].dt.minute / 60
        
        # Create daily summary
        daily = sleep_df.groupby('start_date').agg({
            'sleep_time': 'sum',
            'efficiency': 'mean',
            'fragmentation': 'mean',
            'wake_time': 'mean',
            'onset_latency': 'mean',
            'bedtime_hour': 'mean',
            'waketime_hour': 'mean'
        }).reset_index()
        
        # Store subject info
        subject_info[subject_id] = {
            'n_days': len(daily),
            'daily_data': daily,
            'full_data': sleep_df
        }
        
        all_data.append({
            'subject_id': subject_id,
            'n_days': len(daily)
        })
        
    except Exception as e:
        print(f"  ⚠ Error loading {subject_id}: {e}")
        continue

print(f"Loaded {len(subject_info)} subjects")
print(f"Days per subject: {np.mean([info['n_days'] for info in subject_info.values()]):.1f} ± "
      f"{np.std([info['n_days'] for info in subject_info.values()]):.1f}")
print()

# Extract age from subject_id
def extract_age(subject_id):
    try:
        parts = subject_id.split('_')
        age_part = [p for p in parts if 'mos' in p.lower()]
        if age_part:
            age_str = age_part[0].replace('mos', '').replace('mo', '')
            return int(age_str)
        return None
    except:
        return None

# ============================================================================
# ANALYSIS 1: REGULARITY METRICS (Interdaily Stability & Intradaily Variability)
# ============================================================================

print("Calculating regularity metrics (IS/IV)...")

def calculate_IS_IV(daily_data, variable='bedtime_hour'):
    """
    Calculate Interdaily Stability (IS) and Intradaily Variability (IV)
    
    IS: How stable is the rhythm day-to-day (0-1, higher = more stable)
    IV: How fragmented is the rhythm (0-2, higher = more fragmented)
    """
    if len(daily_data) < 3:
        return None, None
    
    x = daily_data[variable].values
    
    # Remove NaN
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None, None
    
    n = len(x)
    x_mean = np.mean(x)
    
    # Interdaily Stability (IS)
    # Ratio of variance of daily means to total variance
    try:
        # For hourly data, we'd calculate hourly means, but for daily bedtime:
        # IS = variance(daily means) / variance(all data)
        # Since we only have one value per day, we use a sliding window approach
        
        # Simplified IS: consistency of values
        variance_total = np.var(x)
        if variance_total == 0:
            IS = 1.0
        else:
            # Calculate variance of means (pseudo-hourly by treating each day as a bin)
            IS = 1 - (variance_total / (np.var(x) + 1e-10))
            IS = max(0, min(1, IS))  # Clamp to [0, 1]
            
            # Better approach: autocorrelation-based IS
            # IS measures how similar each day is to the mean pattern
            squared_diff_from_mean = np.sum((x - x_mean)**2)
            if squared_diff_from_mean == 0:
                IS = 1.0
            else:
                IS = 1 - (variance_total / (squared_diff_from_mean / n + 1e-10))
                IS = max(0, min(1, IS))
    except:
        IS = None
    
    # Intradaily Variability (IV)
    # Rate of transitions (hour-to-hour changes)
    # For daily data: day-to-day changes
    try:
        diffs = np.diff(x)
        variance_diffs = np.var(diffs)
        variance_total = np.var(x)
        
        if variance_total == 0:
            IV = 0.0
        else:
            IV = variance_diffs / (variance_total + 1e-10)
            IV = max(0, min(2, IV))
    except:
        IV = None
    
    return IS, IV

regularity_results = []

for subject_id, info in subject_info.items():
    daily_data = info['daily_data']
    
    # Calculate IS/IV for bedtime
    IS_bed, IV_bed = calculate_IS_IV(daily_data, 'bedtime_hour')
    
    # Calculate IS/IV for sleep duration
    IS_dur, IV_dur = calculate_IS_IV(daily_data, 'sleep_time')
    
    # Calculate IS/IV for efficiency
    IS_eff, IV_eff = calculate_IS_IV(daily_data, 'efficiency')
    
    age = extract_age(subject_id)
    
    regularity_results.append({
        'subject_id': subject_id,
        'age_months': age,
        'n_days': info['n_days'],
        'IS_bedtime': IS_bed,
        'IV_bedtime': IV_bed,
        'IS_duration': IS_dur,
        'IV_duration': IV_dur,
        'IS_efficiency': IS_eff,
        'IV_efficiency': IV_eff
    })

regularity_df = pd.DataFrame(regularity_results)
regularity_df = regularity_df[regularity_df['age_months'].notna()]

print(f"  Calculated regularity for {len(regularity_df)} subjects")

# ============================================================================
# ANALYSIS 2: RETURN MAP ANALYSIS
# ============================================================================

print("Calculating return map coordinates...")

return_map_data = []

for subject_id, info in subject_info.items():
    daily_data = info['daily_data'].sort_values('start_date')
    
    if len(daily_data) < 2:
        continue
    
    age = extract_age(subject_id)
    
    # Bedtime return map
    bedtimes = daily_data['bedtime_hour'].values
    for i in range(len(bedtimes) - 1):
        if not np.isnan(bedtimes[i]) and not np.isnan(bedtimes[i+1]):
            return_map_data.append({
                'subject_id': subject_id,
                'age_months': age,
                'variable': 'bedtime',
                'value_n': bedtimes[i],
                'value_n1': bedtimes[i+1],
                'deviation': abs(bedtimes[i+1] - bedtimes[i])
            })
    
    # Duration return map
    durations = daily_data['sleep_time'].values
    for i in range(len(durations) - 1):
        if not np.isnan(durations[i]) and not np.isnan(durations[i+1]):
            return_map_data.append({
                'subject_id': subject_id,
                'age_months': age,
                'variable': 'duration',
                'value_n': durations[i],
                'value_n1': durations[i+1],
                'deviation': abs(durations[i+1] - durations[i])
            })
    
    # Wake time return map
    waketimes = daily_data['waketime_hour'].values
    for i in range(len(waketimes) - 1):
        if not np.isnan(waketimes[i]) and not np.isnan(waketimes[i+1]):
            return_map_data.append({
                'subject_id': subject_id,
                'age_months': age,
                'variable': 'waketime',
                'value_n': waketimes[i],
                'value_n1': waketimes[i+1],
                'deviation': abs(waketimes[i+1] - waketimes[i])
            })

return_map_df = pd.DataFrame(return_map_data)
return_map_df = return_map_df[return_map_df['age_months'].notna()]

print(f"  Calculated {len(return_map_df)} return map points")

# ============================================================================
# ANALYSIS 3: AUTOCORRELATION
# ============================================================================

print("Calculating autocorrelation...")

autocorr_results = []

for subject_id, info in subject_info.items():
    daily_data = info['daily_data'].sort_values('start_date')
    
    if len(daily_data) < 4:
        continue
    
    age = extract_age(subject_id)
    
    # Autocorrelation for sleep duration
    duration = daily_data['sleep_time'].dropna().values
    if len(duration) >= 4:
        # Lag 1, 2, 3
        for lag in [1, 2, 3]:
            if len(duration) > lag:
                corr = np.corrcoef(duration[:-lag], duration[lag:])[0, 1]
                autocorr_results.append({
                    'subject_id': subject_id,
                    'age_months': age,
                    'variable': 'duration',
                    'lag': lag,
                    'autocorrelation': corr
                })
    
    # Autocorrelation for efficiency
    efficiency = daily_data['efficiency'].dropna().values
    if len(efficiency) >= 4:
        for lag in [1, 2, 3]:
            if len(efficiency) > lag:
                corr = np.corrcoef(efficiency[:-lag], efficiency[lag:])[0, 1]
                autocorr_results.append({
                    'subject_id': subject_id,
                    'age_months': age,
                    'variable': 'efficiency',
                    'lag': lag,
                    'autocorrelation': corr
                })

autocorr_df = pd.DataFrame(autocorr_results)
autocorr_df = autocorr_df[autocorr_df['age_months'].notna()]

print(f"  Calculated autocorrelation for {len(autocorr_df['subject_id'].unique())} subjects")

# ============================================================================
# ANALYSIS 4: WITHIN-SUBJECT VARIABILITY
# ============================================================================

print("Calculating within-subject variability...")

variability_results = []

for subject_id, info in subject_info.items():
    daily_data = info['daily_data']
    
    if len(daily_data) < 3:
        continue
    
    age = extract_age(subject_id)
    
    # Calculate within-subject SD and CV for key metrics
    duration_mean = daily_data['sleep_time'].mean()
    duration_sd = daily_data['sleep_time'].std()
    duration_cv = (duration_sd / duration_mean * 100) if duration_mean > 0 else None
    
    efficiency_mean = daily_data['efficiency'].mean()
    efficiency_sd = daily_data['efficiency'].std()
    efficiency_cv = (efficiency_sd / efficiency_mean * 100) if efficiency_mean > 0 else None
    
    bedtime_sd = daily_data['bedtime_hour'].std()
    waketime_sd = daily_data['waketime_hour'].std()
    
    variability_results.append({
        'subject_id': subject_id,
        'age_months': age,
        'n_days': len(daily_data),
        'duration_mean': duration_mean,
        'duration_sd': duration_sd,
        'duration_cv': duration_cv,
        'efficiency_mean': efficiency_mean,
        'efficiency_sd': efficiency_sd,
        'efficiency_cv': efficiency_cv,
        'bedtime_sd': bedtime_sd,
        'waketime_sd': waketime_sd,
        'bedtime_sd_minutes': bedtime_sd * 60,
        'waketime_sd_minutes': waketime_sd * 60
    })

variability_df = pd.DataFrame(variability_results)
variability_df = variability_df[variability_df['age_months'].notna()]

print(f"  Calculated variability for {len(variability_df)} subjects")

# ============================================================================
# ANALYSIS 5: SLEEP PHENOTYPE CLUSTERING
# ============================================================================

print("Performing clustering analysis...")

# Prepare features for clustering
cluster_features = variability_df[[
    'duration_mean', 'efficiency_mean', 'duration_cv', 
    'efficiency_cv', 'bedtime_sd_minutes'
]].copy()

# Remove rows with NaN
cluster_features = cluster_features.dropna()
cluster_subjects = variability_df.loc[cluster_features.index, 'subject_id'].values

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(cluster_features)

# K-means clustering (try 3 clusters)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster labels
cluster_df = pd.DataFrame({
    'subject_id': cluster_subjects,
    'cluster': clusters
})

# Merge with original data
variability_df = variability_df.merge(cluster_df, on='subject_id', how='left')

# Calculate cluster profiles
cluster_profiles = []
for i in range(n_clusters):
    cluster_data = variability_df[variability_df['cluster'] == i]
    profile = {
        'cluster': i,
        'n_subjects': len(cluster_data),
        'duration_mean': cluster_data['duration_mean'].mean(),
        'efficiency_mean': cluster_data['efficiency_mean'].mean(),
        'duration_cv': cluster_data['duration_cv'].mean(),
        'bedtime_sd_min': cluster_data['bedtime_sd_minutes'].mean()
    }
    cluster_profiles.append(profile)

cluster_profile_df = pd.DataFrame(cluster_profiles)

print(f"  Identified {n_clusters} sleep phenotypes")
for _, row in cluster_profile_df.iterrows():
    print(f"    Cluster {int(row['cluster'])}: {int(row['n_subjects'])} subjects, "
          f"Duration={row['duration_mean']/60:.1f}h, Efficiency={row['efficiency_mean']:.1f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSaving results...")

regularity_df.to_csv(f'{output_folder}/regularity_metrics.csv', index=False)
return_map_df.to_csv(f'{output_folder}/return_map_data.csv', index=False)
autocorr_df.to_csv(f'{output_folder}/autocorrelation_data.csv', index=False)
variability_df.to_csv(f'{output_folder}/within_subject_variability.csv', index=False)
cluster_profile_df.to_csv(f'{output_folder}/sleep_phenotype_profiles.csv', index=False)

print(f"  ✓ Saved: regularity_metrics.csv")
print(f"  ✓ Saved: return_map_data.csv")
print(f"  ✓ Saved: autocorrelation_data.csv")
print(f"  ✓ Saved: within_subject_variability.csv")
print(f"  ✓ Saved: sleep_phenotype_profiles.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

# Get age groups
age_groups = sorted(regularity_df['age_months'].unique())
colors_age = {16: '#A23B72', 21: '#F18F01', 26: '#2E86AB', 31: '#C73E1D'}

# ============================================================================
# FIGURE: Regularity Metrics (IS/IV)
# ============================================================================

print("  Creating regularity metrics plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: IS vs IV for Bedtime
ax = axes[0, 0]
for age in age_groups:
    age_data = regularity_df[regularity_df['age_months'] == age]
    ax.scatter(age_data['IS_bedtime'], age_data['IV_bedtime'],
              s=100, alpha=0.6, color=colors_age.get(age, 'gray'),
              label=f'{age} months', edgecolors='white', linewidth=1.5)

ax.set_xlabel('Interdaily Stability (IS)', fontsize=12, fontweight='bold')
ax.set_ylabel('Intradaily Variability (IV)', fontsize=12, fontweight='bold')
ax.set_title('Bedtime Regularity: IS vs IV', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add quadrant labels
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
ax.text(0.75, 0.25, 'Good\nRegularity', transform=ax.transAxes,
       ha='center', fontsize=9, alpha=0.5)

# Panel B: IS by Age
ax = axes[0, 1]
is_means = regularity_df.groupby('age_months')['IS_bedtime'].mean()
is_stds = regularity_df.groupby('age_months')['IS_bedtime'].std()

ax.errorbar(age_groups, is_means, yerr=is_stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB', label='Bedtime IS')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Interdaily Stability', fontsize=12, fontweight='bold')
ax.set_title('Circadian Rhythm Stability by Age', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel C: IV by Age
ax = axes[1, 0]
iv_means = regularity_df.groupby('age_months')['IV_bedtime'].mean()
iv_stds = regularity_df.groupby('age_months')['IV_bedtime'].std()

ax.errorbar(age_groups, iv_means, yerr=iv_stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#F18F01', label='Bedtime IV')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Intradaily Variability', fontsize=12, fontweight='bold')
ax.set_title('Sleep Fragmentation by Age', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel D: Individual IS scores
ax = axes[1, 1]
for age in age_groups:
    age_data = regularity_df[regularity_df['age_months'] == age]
    x_pos = [age] * len(age_data)
    ax.scatter(x_pos, age_data['IS_bedtime'],
              s=80, alpha=0.5, color=colors_age.get(age, 'gray'))

# Overlay means
ax.plot(age_groups, is_means, 'ko-', markersize=12, linewidth=3,
       label='Mean', zorder=10)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Interdaily Stability (IS)', fontsize=12, fontweight='bold')
ax.set_title('Individual Regularity Scores', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f'{output_folder}/dynamics_fig1_regularity_metrics.png', dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: dynamics_fig1_regularity_metrics.png")
plt.close()

# ============================================================================
# FIGURE: Return Maps
# ============================================================================

print("  Creating return map plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Bedtime Return Map (All Ages)
ax = axes[0, 0]
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'bedtime')]
    ax.scatter(age_data['value_n'], age_data['value_n1'],
              s=30, alpha=0.4, color=colors_age.get(age, 'gray'),
              label=f'{age} months')

# Add diagonal (perfect stability)
ax.plot([18, 24], [18, 24], 'k--', linewidth=2, alpha=0.5, label='Perfect stability')

ax.set_xlabel('Bedtime Day N (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Bedtime Day N+1 (hours)', fontsize=12, fontweight='bold')
ax.set_title('Bedtime Return Map: Day-to-Day Stability', fontsize=13, fontweight='bold')

# Format axes
x_ticks = [18, 19, 20, 21, 22, 23, 24]
x_labels = ['6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM', '12 AM']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45)
ax.set_yticks(x_ticks)
ax.set_yticklabels(x_labels)

ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: Duration Return Map
ax = axes[0, 1]
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'duration')]
    ax.scatter(age_data['value_n'] / 60, age_data['value_n1'] / 60,
              s=30, alpha=0.4, color=colors_age.get(age, 'gray'),
              label=f'{age} months')

# Add diagonal
min_dur = return_map_df[return_map_df['variable'] == 'duration']['value_n'].min() / 60
max_dur = return_map_df[return_map_df['variable'] == 'duration']['value_n'].max() / 60
ax.plot([min_dur, max_dur], [min_dur, max_dur], 'k--', linewidth=2, alpha=0.5)

ax.set_xlabel('Sleep Duration Day N (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Duration Day N+1 (hours)', fontsize=12, fontweight='bold')
ax.set_title('Duration Return Map: Homeostatic Regulation', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: Wake Time Return Map
ax = axes[1, 0]
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'waketime')]
    ax.scatter(age_data['value_n'], age_data['value_n1'],
              s=30, alpha=0.4, color=colors_age.get(age, 'gray'),
              label=f'{age} months')

# Add diagonal
ax.plot([5, 10], [5, 10], 'k--', linewidth=2, alpha=0.5)

ax.set_xlabel('Wake Time Day N (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wake Time Day N+1 (hours)', fontsize=12, fontweight='bold')
ax.set_title('Wake Time Return Map', fontsize=13, fontweight='bold')

y_ticks = [5, 6, 7, 8, 9, 10]
y_labels = ['5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM']
ax.set_xticks(y_ticks)
ax.set_xticklabels(y_labels, rotation=45)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel D: Deviation from Diagonal (Stability Metric)
ax = axes[1, 1]

bedtime_deviations = []
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'bedtime')]
    mean_dev = age_data['deviation'].mean() * 60  # Convert to minutes
    bedtime_deviations.append(mean_dev)

ax.plot(age_groups, bedtime_deviations, marker='o', markersize=10, 
       linewidth=2, color='#2E86AB', label='Bedtime')

# Add duration deviations
duration_deviations = []
for age in age_groups:
    age_data = return_map_df[(return_map_df['age_months'] == age) & 
                             (return_map_df['variable'] == 'duration')]
    mean_dev = age_data['deviation'].mean()  # Already in minutes
    duration_deviations.append(mean_dev)

ax.plot(age_groups, duration_deviations, marker='s', markersize=10,
       linewidth=2, color='#F18F01', label='Duration')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Day-to-Day Change (minutes)', fontsize=12, fontweight='bold')
ax.set_title('System Stability: Day-to-Day Variability', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/dynamics_fig2_return_maps.png', dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: dynamics_fig2_return_maps.png")
plt.close()

# ============================================================================
# FIGURE: Autocorrelation
# ============================================================================

print("  Creating autocorrelation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Duration Autocorrelation by Age
ax = axes[0, 0]
for age in age_groups:
    age_data = autocorr_df[(autocorr_df['age_months'] == age) & 
                           (autocorr_df['variable'] == 'duration')]
    
    lag_means = age_data.groupby('lag')['autocorrelation'].mean()
    lags = lag_means.index.tolist()
    
    ax.plot(lags, lag_means, marker='o', markersize=8, linewidth=2,
           color=colors_age.get(age, 'gray'), label=f'{age} months')

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax.set_title('Sleep Duration Autocorrelation', fontsize=13, fontweight='bold')
ax.set_xticks([1, 2, 3])
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Efficiency Autocorrelation by Age
ax = axes[0, 1]
for age in age_groups:
    age_data = autocorr_df[(autocorr_df['age_months'] == age) & 
                           (autocorr_df['variable'] == 'efficiency')]
    
    lag_means = age_data.groupby('lag')['autocorrelation'].mean()
    lags = lag_means.index.tolist()
    
    ax.plot(lags, lag_means, marker='s', markersize=8, linewidth=2,
           color=colors_age.get(age, 'gray'), label=f'{age} months')

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax.set_title('Sleep Efficiency Autocorrelation', fontsize=13, fontweight='bold')
ax.set_xticks([1, 2, 3])
ax.legend()
ax.grid(True, alpha=0.3)

# Panel C: Lag-1 Autocorrelation by Age (Both Variables)
ax = axes[1, 0]

duration_lag1 = []
efficiency_lag1 = []

for age in age_groups:
    dur_data = autocorr_df[(autocorr_df['age_months'] == age) & 
                           (autocorr_df['variable'] == 'duration') &
                           (autocorr_df['lag'] == 1)]
    duration_lag1.append(dur_data['autocorrelation'].mean())
    
    eff_data = autocorr_df[(autocorr_df['age_months'] == age) & 
                           (autocorr_df['variable'] == 'efficiency') &
                           (autocorr_df['lag'] == 1)]
    efficiency_lag1.append(eff_data['autocorrelation'].mean())

x = np.arange(len(age_groups))
width = 0.35

ax.bar(x - width/2, duration_lag1, width, label='Duration', 
      color='#2E86AB', alpha=0.8)
ax.bar(x + width/2, efficiency_lag1, width, label='Efficiency',
      color='#F18F01', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Lag-1 Autocorrelation', fontsize=12, fontweight='bold')
ax.set_title('Sleep Memory Effect (1-Day Lag)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Individual Autocorrelation Distribution
ax = axes[1, 1]

lag1_duration = autocorr_df[(autocorr_df['lag'] == 1) & 
                            (autocorr_df['variable'] == 'duration')]

data_list = [lag1_duration[lag1_duration['age_months'] == age]['autocorrelation'].dropna() 
            for age in age_groups]

bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('#2E86AB')
    patch.set_alpha(0.6)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Duration Autocorrelation (Lag 1)', fontsize=12, fontweight='bold')
ax.set_title('Individual Variability in Sleep Memory', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_folder}/dynamics_fig3_autocorrelation.png', dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: dynamics_fig3_autocorrelation.png")
plt.close()

# ============================================================================
# FIGURE: Within-Subject Variability
# ============================================================================

print("  Creating variability analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Within-Subject CV by Age
ax = axes[0, 0]

cv_means_dur = variability_df.groupby('age_months')['duration_cv'].mean()
cv_stds_dur = variability_df.groupby('age_months')['duration_cv'].std()

cv_means_eff = variability_df.groupby('age_months')['efficiency_cv'].mean()
cv_stds_eff = variability_df.groupby('age_months')['efficiency_cv'].std()

ax.errorbar(age_groups, cv_means_dur, yerr=cv_stds_dur,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#2E86AB', label='Duration CV')
ax.errorbar(age_groups, cv_means_eff, yerr=cv_stds_eff,
           marker='s', markersize=10, linewidth=2, capsize=5,
           color='#F18F01', label='Efficiency CV')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax.set_title('Within-Subject Variability by Age', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Bedtime Consistency
ax = axes[0, 1]

bedtime_sd_means = variability_df.groupby('age_months')['bedtime_sd_minutes'].mean()
bedtime_sd_stds = variability_df.groupby('age_months')['bedtime_sd_minutes'].std()

ax.errorbar(age_groups, bedtime_sd_means, yerr=bedtime_sd_stds,
           marker='o', markersize=10, linewidth=2, capsize=5,
           color='#A23B72')

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Within-Subject Bedtime SD (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Bedtime Consistency by Age', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)

# Panel C: Individual Variability Profiles
ax = axes[1, 0]

for age in age_groups:
    age_data = variability_df[variability_df['age_months'] == age]
    ax.scatter([age] * len(age_data), age_data['duration_cv'],
              s=80, alpha=0.5, color=colors_age.get(age, 'gray'))

ax.plot(age_groups, cv_means_dur, 'ko-', markersize=12, linewidth=3,
       label='Mean', zorder=10)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Duration CV (%)', fontsize=12, fontweight='bold')
ax.set_title('Individual Sleep Variability', fontsize=13, fontweight='bold')
ax.set_xticks(age_groups)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: CV vs IS (Regularity)
ax = axes[1, 1]

merged = variability_df.merge(regularity_df[['subject_id', 'IS_bedtime']], 
                              on='subject_id', how='inner')

for age in age_groups:
    age_data = merged[merged['age_months'] == age]
    ax.scatter(age_data['IS_bedtime'], age_data['duration_cv'],
              s=100, alpha=0.6, color=colors_age.get(age, 'gray'),
              label=f'{age} months', edgecolors='white', linewidth=1.5)

ax.set_xlabel('Interdaily Stability (IS)', fontsize=12, fontweight='bold')
ax.set_ylabel('Duration CV (%)', fontsize=12, fontweight='bold')
ax.set_title('Regularity vs Variability', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/dynamics_fig4_within_subject_variability.png', 
           dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: dynamics_fig4_within_subject_variability.png")
plt.close()

# ============================================================================
# FIGURE: Sleep Phenotype Clustering
# ============================================================================

print("  Creating clustering analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Cluster Profiles (Radar Chart)
ax = axes[0, 0]

# Prepare data for radar chart
categories = ['Duration\n(hours)', 'Efficiency\n(%)', 'Duration\nVariability', 
             'Bedtime\nConsistency']

cluster_colors = ['#2E86AB', '#F18F01', '#A23B72']

# Normalize values for radar chart
duration_norm = cluster_profile_df['duration_mean'] / 60 / 12  # Normalize to 12h
efficiency_norm = cluster_profile_df['efficiency_mean'] / 100
cv_norm = 1 - (cluster_profile_df['duration_cv'] / cluster_profile_df['duration_cv'].max())  # Invert
bedtime_norm = 1 - (cluster_profile_df['bedtime_sd_min'] / cluster_profile_df['bedtime_sd_min'].max())  # Invert

# Create bar chart instead of radar (simpler)
x = np.arange(len(categories))
width = 0.25

for i in range(n_clusters):
    values = [
        duration_norm.iloc[i],
        efficiency_norm.iloc[i],
        cv_norm.iloc[i],
        bedtime_norm.iloc[i]
    ]
    ax.bar(x + i*width, values, width, label=f'Phenotype {i+1}',
          color=cluster_colors[i], alpha=0.8)

ax.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
ax.set_title('Sleep Phenotype Profiles', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(categories, fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Cluster Distribution by Age
ax = axes[0, 1]

cluster_age_counts = variability_df.groupby(['age_months', 'cluster']).size().unstack(fill_value=0)

cluster_age_counts.plot(kind='bar', stacked=True, ax=ax, 
                       color=cluster_colors, alpha=0.8)

ax.set_xlabel('Age (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
ax.set_title('Sleep Phenotypes by Age', fontsize=13, fontweight='bold')
ax.legend(title='Phenotype', labels=[f'Phenotype {i+1}' for i in range(n_clusters)])
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

# Panel C: Cluster Scatter (Duration vs Efficiency)
ax = axes[1, 0]

for i in range(n_clusters):
    cluster_data = variability_df[variability_df['cluster'] == i]
    ax.scatter(cluster_data['duration_mean'] / 60, 
              cluster_data['efficiency_mean'],
              s=100, alpha=0.6, color=cluster_colors[i],
              label=f'Phenotype {i+1}', edgecolors='white', linewidth=1.5)

ax.set_xlabel('Mean Sleep Duration (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Sleep Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('Sleep Phenotypes: Duration vs Efficiency', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: Phenotype Characteristics Table
ax = axes[1, 1]
ax.axis('off')

# Create summary text
summary_text = "Sleep Phenotype Characteristics:\n\n"

for i in range(n_clusters):
    cluster_data = variability_df[variability_df['cluster'] == i]
    profile = cluster_profile_df[cluster_profile_df['cluster'] == i].iloc[0]
    
    summary_text += f"Phenotype {i+1} (n={int(profile['n_subjects'])}):\n"
    summary_text += f"  Duration: {profile['duration_mean']/60:.1f} hours\n"
    summary_text += f"  Efficiency: {profile['efficiency_mean']:.1f}%\n"
    summary_text += f"  Variability: {profile['duration_cv']:.1f}% CV\n"
    summary_text += f"  Bedtime SD: ±{profile['bedtime_sd_min']:.0f} min\n"
    
    # Interpretation
    if profile['efficiency_mean'] > 90 and profile['duration_cv'] < 10:
        summary_text += "  → High-quality, stable sleepers\n"
    elif profile['efficiency_mean'] < 85:
        summary_text += "  → Lower efficiency sleepers\n"
    elif profile['duration_cv'] > 15:
        summary_text += "  → Variable sleep patterns\n"
    
    summary_text += "\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

plt.tight_layout()
plt.savefig(f'{output_folder}/dynamics_fig5_sleep_phenotypes.png', 
           dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: dynamics_fig5_sleep_phenotypes.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ADVANCED DYNAMICS ANALYSIS COMPLETE")
print("="*70)

print("\nKey Findings:")
print(f"  • Mean Interdaily Stability (IS): {regularity_df['IS_bedtime'].mean():.3f}")
print(f"  • Mean Intradaily Variability (IV): {regularity_df['IV_bedtime'].mean():.3f}")
print(f"  • Mean Day-to-Day Bedtime Change: {return_map_df[return_map_df['variable']=='bedtime']['deviation'].mean()*60:.1f} minutes")
print(f"  • Lag-1 Duration Autocorrelation: {autocorr_df[(autocorr_df['lag']==1) & (autocorr_df['variable']=='duration')]['autocorrelation'].mean():.3f}")
print(f"  • {n_clusters} distinct sleep phenotypes identified")

print("\nFiles saved to:", output_folder)
print("  - 5 CSV data files")
print("  - 5 visualization figures")

print("="*70)