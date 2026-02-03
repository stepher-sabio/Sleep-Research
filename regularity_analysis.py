"""
regularity_analysis.py
Circadian rhythm regularity analysis and sleep phenotype clustering

Analyses:
- Interdaily Stability (IS): Day-to-day rhythm consistency
- Intradaily Variability (IV): Within-day fragmentation
- Sleep phenotype identification through clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/age_groups'
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("REGULARITY & PHENOTYPE ANALYSIS")
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
        for col in ['sleep_time', 'efficiency', 'fragmentation']:
            if col in sleep_df.columns:
                sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
        
        sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
        
        if len(sleep_df) == 0:
            continue
        
        # Parse times
        sleep_df['bedtime'] = pd.to_datetime(sleep_df['start_date'] + ' ' + sleep_df['start_time'])
        sleep_df['bedtime_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
        
        # Daily aggregation
        daily = sleep_df.groupby('start_date').agg({
            'sleep_time': 'sum',
            'efficiency': 'mean',
            'bedtime_hour': 'mean'
        }).reset_index()
        
        if len(daily) >= 3:  # Need at least 3 days
            subject_data[subject_id] = daily
        
    except Exception as e:
        continue

print(f"Loaded {len(subject_data)} subjects with 3+ days of data\n")

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
# REGULARITY METRICS (IS/IV)
# ============================================================================

print("Calculating regularity metrics...")

def calculate_IS(values):
    """Interdaily Stability: rhythm consistency (0-1, higher = more stable)"""
    if len(values) < 3:
        return None
    
    # Remove NaN
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return None
    
    # IS based on variance decomposition
    mean_val = np.mean(values)
    total_var = np.var(values)
    
    if total_var == 0:
        return 1.0
    
    # Calculate stability as inverse of coefficient of variation
    cv = np.std(values) / (mean_val + 1e-10)
    IS = 1 / (1 + cv)  # Transform to 0-1 scale
    
    return min(1.0, max(0.0, IS))

def calculate_IV(values):
    """Intradaily Variability: day-to-day changes (0-2, higher = more variable)"""
    if len(values) < 3:
        return None
    
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return None
    
    # IV based on successive differences
    diffs = np.diff(values)
    var_diffs = np.var(diffs)
    var_total = np.var(values)
    
    if var_total == 0:
        return 0.0
    
    IV = var_diffs / (var_total + 1e-10)
    return min(2.0, max(0.0, IV))

regularity_results = []

for subject_id, daily_data in subject_data.items():
    age = extract_age(subject_id)
    
    # Bedtime regularity
    IS_bed = calculate_IS(daily_data['bedtime_hour'].values)
    IV_bed = calculate_IV(daily_data['bedtime_hour'].values)
    
    # Duration regularity
    IS_dur = calculate_IS(daily_data['sleep_time'].values)
    IV_dur = calculate_IV(daily_data['sleep_time'].values)
    
    # Efficiency regularity
    IS_eff = calculate_IS(daily_data['efficiency'].values)
    IV_eff = calculate_IV(daily_data['efficiency'].values)
    
    # Variability metrics
    duration_mean = daily_data['sleep_time'].mean()
    duration_sd = daily_data['sleep_time'].std()
    duration_cv = (duration_sd / duration_mean * 100) if duration_mean > 0 else None
    
    efficiency_mean = daily_data['efficiency'].mean()
    bedtime_sd = daily_data['bedtime_hour'].std() * 60  # Convert to minutes
    
    regularity_results.append({
        'subject_id': subject_id,
        'age_months': age,
        'n_days': len(daily_data),
        'IS_bedtime': IS_bed,
        'IV_bedtime': IV_bed,
        'IS_duration': IS_dur,
        'IV_duration': IV_dur,
        'IS_efficiency': IS_eff,
        'IV_efficiency': IV_eff,
        'duration_mean': duration_mean,
        'duration_cv': duration_cv,
        'efficiency_mean': efficiency_mean,
        'bedtime_sd_minutes': bedtime_sd
    })

regularity_df = pd.DataFrame(regularity_results)
regularity_df = regularity_df[regularity_df['age_months'].notna()]

print(f"Calculated regularity for {len(regularity_df)} subjects\n")

# Save
regularity_df.to_csv(f'{output_folder}/regularity_metrics_detailed.csv', index=False)
print(f"✓ Saved: regularity_metrics_detailed.csv\n")

# ============================================================================
# CLUSTERING ANALYSIS (Sleep Phenotypes)
# ============================================================================

print("Performing clustering analysis...")

# Prepare features
cluster_features = regularity_df[[
    'duration_mean', 'efficiency_mean', 'duration_cv', 
    'bedtime_sd_minutes', 'IS_bedtime'
]].copy()

# Remove NaN
cluster_features = cluster_features.dropna()
cluster_subjects = regularity_df.loc[cluster_features.index, 'subject_id'].values
cluster_ages = regularity_df.loc[cluster_features.index, 'age_months'].values

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(cluster_features)

# K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Create cluster dataframe
cluster_df = pd.DataFrame({
    'subject_id': cluster_subjects,
    'age_months': cluster_ages,
    'cluster': clusters
})

# Merge with regularity data
regularity_df = regularity_df.merge(cluster_df[['subject_id', 'cluster']], 
                                    on='subject_id', how='left')

# Calculate cluster profiles
cluster_profiles = []
for i in range(n_clusters):
    cluster_data = regularity_df[regularity_df['cluster'] == i]
    
    profile = {
        'cluster': i,
        'n_subjects': len(cluster_data),
        'duration_mean_hours': cluster_data['duration_mean'].mean() / 60,
        'efficiency_mean': cluster_data['efficiency_mean'].mean(),
        'duration_cv': cluster_data['duration_cv'].mean(),
        'bedtime_sd_min': cluster_data['bedtime_sd_minutes'].mean(),
        'IS_bedtime': cluster_data['IS_bedtime'].mean(),
        'IV_bedtime': cluster_data['IV_bedtime'].mean()
    }
    
    # Assign label based on characteristics
    if profile['efficiency_mean'] > 90 and profile['duration_cv'] < 10:
        profile['label'] = 'High-Quality Stable'
    elif profile['duration_cv'] > 15:
        profile['label'] = 'Variable Sleepers'
    elif profile['efficiency_mean'] < 85:
        profile['label'] = 'Lower Efficiency'
    else:
        profile['label'] = 'Moderate'
    
    cluster_profiles.append(profile)

cluster_profile_df = pd.DataFrame(cluster_profiles)

print(f"Identified {n_clusters} sleep phenotypes:")
for _, row in cluster_profile_df.iterrows():
    print(f"  Cluster {int(row['cluster'])} ({row['label']}): {int(row['n_subjects'])} subjects")
    print(f"    Duration: {row['duration_mean_hours']:.1f}h, Efficiency: {row['efficiency_mean']:.1f}%")

# Save
cluster_profile_df.to_csv(f'{output_folder}/sleep_phenotype_profiles_detailed.csv', index=False)
print(f"\n✓ Saved: sleep_phenotype_profiles_detailed.csv\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")

age_groups = sorted(regularity_df['age_months'].unique())
colors_age = {16: '#A23B72', 21: '#F18F01', 26: '#2E86AB', 31: '#C73E1D'}
colors_cluster = ['#2E86AB', '#F18F01', '#A23B72']

# ============================================================================
# Figure 1: Regularity Metrics Overview
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# IS by Age
ax = axes[0, 0]
is_means = regularity_df.groupby('age_months')['IS_bedtime'].mean()
is_stds = regularity_df.groupby('age_months')['IS_bedtime'].std()
ax.errorbar(age_groups, is_means, yerr=is_stds, marker='o', markersize=10,
           linewidth=2, capsize=5, color='#2E86AB')
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Interdaily Stability', fontweight='bold')
ax.set_title('Circadian Rhythm Stability', fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)

# IV by Age
ax = axes[0, 1]
iv_means = regularity_df.groupby('age_months')['IV_bedtime'].mean()
iv_stds = regularity_df.groupby('age_months')['IV_bedtime'].std()
ax.errorbar(age_groups, iv_means, yerr=iv_stds, marker='o', markersize=10,
           linewidth=2, capsize=5, color='#F18F01')
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Intradaily Variability', fontweight='bold')
ax.set_title('Day-to-Day Variability', fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)

# IS vs IV Scatter
ax = axes[0, 2]
for age in age_groups:
    age_data = regularity_df[regularity_df['age_months'] == age]
    ax.scatter(age_data['IS_bedtime'], age_data['IV_bedtime'],
              s=100, alpha=0.6, color=colors_age.get(age, 'gray'),
              label=f'{age} months', edgecolors='white', linewidth=1.5)
ax.set_xlabel('Interdaily Stability (IS)', fontweight='bold')
ax.set_ylabel('Intradaily Variability (IV)', fontweight='bold')
ax.set_title('Regularity Profile', fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# IS Distribution by Age
ax = axes[1, 0]
data_list = [regularity_df[regularity_df['age_months'] == age]['IS_bedtime'].dropna() 
            for age in age_groups]
bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('#2E86AB')
    patch.set_alpha(0.7)
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Interdaily Stability', fontweight='bold')
ax.set_title('IS Distribution by Age', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Regularity Score (Combined IS/IV)
ax = axes[1, 1]
# Higher IS and lower IV = better regularity
regularity_df['regularity_score'] = regularity_df['IS_bedtime'] * (1 - regularity_df['IV_bedtime']/2)
reg_means = regularity_df.groupby('age_months')['regularity_score'].mean()
reg_stds = regularity_df.groupby('age_months')['regularity_score'].std()
ax.errorbar(age_groups, reg_means, yerr=reg_stds, marker='o', markersize=10,
           linewidth=2, capsize=5, color='#A23B72')
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Regularity Score', fontweight='bold')
ax.set_title('Overall Sleep Regularity', fontweight='bold')
ax.set_xticks(age_groups)
ax.grid(True, alpha=0.3)

# Correlation: IS vs Efficiency
ax = axes[1, 2]
for age in age_groups:
    age_data = regularity_df[regularity_df['age_months'] == age]
    ax.scatter(age_data['IS_bedtime'], age_data['efficiency_mean'],
              s=100, alpha=0.6, color=colors_age.get(age, 'gray'),
              edgecolors='white', linewidth=1.5)
ax.set_xlabel('Interdaily Stability', fontweight='bold')
ax.set_ylabel('Sleep Efficiency (%)', fontweight='bold')
ax.set_title('Regularity vs Sleep Quality', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/regularity_fig1_overview.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: regularity_fig1_overview.png")
plt.close()

# ============================================================================
# Figure 2: Sleep Phenotypes
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Phenotype Profiles
ax = axes[0, 0]
x = np.arange(n_clusters)
width = 0.2

metrics_normalized = {
    'Duration': cluster_profile_df['duration_mean_hours'].values / 12,
    'Efficiency': cluster_profile_df['efficiency_mean'].values / 100,
    'Stability': cluster_profile_df['IS_bedtime'].values,
    'Consistency': 1 - cluster_profile_df['duration_cv'].values / 20
}

for i, (metric, values) in enumerate(metrics_normalized.items()):
    ax.bar(x + i*width, values, width, label=metric, alpha=0.8)

ax.set_ylabel('Normalized Score', fontweight='bold')
ax.set_title('Sleep Phenotype Profiles', fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"Phenotype {i+1}" for i in range(n_clusters)])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Phenotype Distribution by Age
ax = axes[0, 1]
cluster_age_counts = regularity_df.groupby(['age_months', 'cluster']).size().unstack(fill_value=0)
cluster_age_counts.plot(kind='bar', stacked=True, ax=ax, color=colors_cluster, alpha=0.8)
ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Number of Subjects', fontweight='bold')
ax.set_title('Phenotype Distribution by Age', fontweight='bold')
ax.legend(title='Phenotype', labels=[f'Phenotype {i+1}' for i in range(n_clusters)])
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

# Duration vs Efficiency by Phenotype
ax = axes[1, 0]
for i in range(n_clusters):
    cluster_data = regularity_df[regularity_df['cluster'] == i]
    ax.scatter(cluster_data['duration_mean'] / 60, cluster_data['efficiency_mean'],
              s=100, alpha=0.6, color=colors_cluster[i],
              label=f'Phenotype {i+1}', edgecolors='white', linewidth=1.5)
ax.set_xlabel('Mean Sleep Duration (hours)', fontweight='bold')
ax.set_ylabel('Mean Sleep Efficiency (%)', fontweight='bold')
ax.set_title('Phenotype Characteristics', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Phenotype Table
ax = axes[1, 1]
ax.axis('off')

table_text = "Sleep Phenotype Summary\n" + "="*40 + "\n\n"
for _, row in cluster_profile_df.iterrows():
    i = int(row['cluster'])
    table_text += f"Phenotype {i+1}: {row['label']}\n"
    table_text += f"  N = {int(row['n_subjects'])} subjects\n"
    table_text += f"  Duration: {row['duration_mean_hours']:.1f} hours\n"
    table_text += f"  Efficiency: {row['efficiency_mean']:.1f}%\n"
    table_text += f"  Variability: {row['duration_cv']:.1f}% CV\n"
    table_text += f"  IS: {row['IS_bedtime']:.3f}\n"
    table_text += f"  Bedtime SD: ±{row['bedtime_sd_min']:.0f} min\n\n"

ax.text(0.1, 0.9, table_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

plt.tight_layout()
plt.savefig(f'{output_folder}/regularity_fig2_phenotypes.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: regularity_fig2_phenotypes.png")
plt.close()

print("\n" + "="*70)
print("REGULARITY ANALYSIS COMPLETE")
print("="*70)
print(f"\nKey Findings:")
print(f"  • Mean IS (bedtime): {regularity_df['IS_bedtime'].mean():.3f}")
print(f"  • Mean IV (bedtime): {regularity_df['IV_bedtime'].mean():.3f}")
print(f"  • {n_clusters} distinct sleep phenotypes identified")
print("\nFiles saved:")
print("  • regularity_metrics_detailed.csv")
print("  • sleep_phenotype_profiles_detailed.csv")
print("  • regularity_fig1_overview.png")
print("  • regularity_fig2_phenotypes.png")
print("="*70)