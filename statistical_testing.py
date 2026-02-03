"""
statistical_testing.py
Statistical hypothesis testing for age group comparisons

Analyses:
- ANOVA for all sleep metrics
- Post-hoc pairwise comparisons
- Effect sizes (Cohen's d)
- Publication-ready tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_csv = '/Users/stepher/Desktop/Actigraphy2/results/all_subjects_summary.csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/age_groups'
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("STATISTICAL HYPOTHESIS TESTING")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading summary data...")

df = pd.read_csv(input_csv)

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

df['age_months'] = df['subject_id'].apply(extract_age)
df = df[df['age_months'].notna()]

print(f"Loaded {len(df)} subjects")
print(f"Age groups: {sorted(df['age_months'].unique())}")
print(f"\nSample sizes:")
print(df['age_months'].value_counts().sort_index())

age_groups = sorted(df['age_months'].unique())

# ============================================================================
# DEFINE METRICS TO TEST
# ============================================================================

metrics_to_test = {
    'Sleep Duration': 'sleep_time_mean',
    'Sleep Efficiency': 'efficiency_mean',
    'Fragmentation': 'fragmentation_mean',
    'Onset Latency': 'onset_latency_mean',
    'WASO': 'wake_time_mean',
    'Evening Duration': 'evening_duration_mean',
    'Sleep Intervals/Day': 'avg_intervals_per_day'
}

# ============================================================================
# ANOVA TESTING
# ============================================================================

print("\n" + "="*70)
print("ANOVA RESULTS")
print("="*70)

anova_results = []

for metric_name, col_name in metrics_to_test.items():
    # Prepare data
    groups = [df[df['age_months'] == age][col_name].dropna().values 
             for age in age_groups]
    
    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        continue
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate eta-squared (effect size for ANOVA)
    # eta^2 = SSbetween / SStotal
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Interpretation
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    anova_results.append({
        'Metric': metric_name,
        'F-statistic': f_stat,
        'p-value': p_value,
        'eta-squared': eta_squared,
        'Significance': sig
    })
    
    print(f"\n{metric_name}:")
    print(f"  F({len(groups)-1}, {len(all_data)-len(groups)}) = {f_stat:.3f}, p = {p_value:.4f} {sig}")
    print(f"  η² = {eta_squared:.3f}")

anova_df = pd.DataFrame(anova_results)

# Save
anova_df.to_csv(f'{output_folder}/anova_results.csv', index=False)
print(f"\n✓ Saved: anova_results.csv")

# ============================================================================
# POST-HOC PAIRWISE COMPARISONS
# ============================================================================

print("\n" + "="*70)
print("POST-HOC PAIRWISE COMPARISONS (t-tests with Bonferroni correction)")
print("="*70)

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

posthoc_results = []

for metric_name, col_name in metrics_to_test.items():
    
    print(f"\n{metric_name}:")
    
    # All pairwise combinations
    pairs = list(combinations(age_groups, 2))
    n_comparisons = len(pairs)
    alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction
    
    for age1, age2 in pairs:
        group1 = df[df['age_months'] == age1][col_name].dropna().values
        group2 = df[df['age_months'] == age2][col_name].dropna().values
        
        if len(group1) < 2 or len(group2) < 2:
            continue
        
        # t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Cohen's d
        d = cohens_d(group1, group2)
        
        # Effect size interpretation
        if abs(d) < 0.2:
            effect_interp = 'negligible'
        elif abs(d) < 0.5:
            effect_interp = 'small'
        elif abs(d) < 0.8:
            effect_interp = 'medium'
        else:
            effect_interp = 'large'
        
        # Significance
        if p_value < alpha_corrected:
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
        else:
            sig = 'ns'
        
        mean_diff = np.mean(group1) - np.mean(group2)
        
        posthoc_results.append({
            'Metric': metric_name,
            'Age Group 1': age1,
            'Age Group 2': age2,
            'Mean Difference': mean_diff,
            't-statistic': t_stat,
            'p-value': p_value,
            'p-value (corrected)': p_value * n_comparisons,
            "Cohen's d": d,
            'Effect Size': effect_interp,
            'Significance': sig
        })
        
        print(f"  {age1} vs {age2}: Δ={mean_diff:.2f}, t={t_stat:.2f}, p={p_value:.4f} {sig}, d={d:.2f} ({effect_interp})")

posthoc_df = pd.DataFrame(posthoc_results)

# Save
posthoc_df.to_csv(f'{output_folder}/posthoc_comparisons.csv', index=False)
print(f"\n✓ Saved: posthoc_comparisons.csv")

# ============================================================================
# DESCRIPTIVE STATISTICS BY AGE GROUP
# ============================================================================

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS BY AGE GROUP")
print("="*70)

descriptive_results = []

for age in age_groups:
    age_data = df[df['age_months'] == age]
    
    stats_row = {'Age (months)': age, 'N': len(age_data)}
    
    for metric_name, col_name in metrics_to_test.items():
        values = age_data[col_name].dropna()
        if len(values) > 0:
            stats_row[f'{metric_name} Mean'] = values.mean()
            stats_row[f'{metric_name} SD'] = values.std()
        else:
            stats_row[f'{metric_name} Mean'] = None
            stats_row[f'{metric_name} SD'] = None
    
    descriptive_results.append(stats_row)

descriptive_df = pd.DataFrame(descriptive_results)

# Save
descriptive_df.to_csv(f'{output_folder}/descriptive_statistics_by_age.csv', index=False)
print(f"✓ Saved: descriptive_statistics_by_age.csv")

print("\n")
print(descriptive_df.to_string(index=False))

# ============================================================================
# VISUALIZATION: Effect Sizes
# ============================================================================

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ANOVA Results
ax = axes[0, 0]
metrics_sig = anova_df.sort_values('p-value')
colors = ['green' if p < 0.05 else 'gray' for p in metrics_sig['p-value']]

ax.barh(metrics_sig['Metric'], -np.log10(metrics_sig['p-value']), color=colors, alpha=0.7)
ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
ax.axvline(x=-np.log10(0.01), color='darkred', linestyle='--', linewidth=2, label='p=0.01')
ax.set_xlabel('-log10(p-value)', fontweight='bold')
ax.set_title('ANOVA Results: Age Effect Significance', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Effect Sizes (eta-squared)
ax = axes[0, 1]
ax.barh(metrics_sig['Metric'], metrics_sig['eta-squared'], color='steelblue', alpha=0.7)
ax.set_xlabel('η² (Effect Size)', fontweight='bold')
ax.set_title('ANOVA Effect Sizes', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Post-hoc Effect Sizes (Cohen's d)
ax = axes[1, 0]

# Focus on sleep duration comparisons
duration_posthoc = posthoc_df[posthoc_df['Metric'] == 'Sleep Duration'].copy()
duration_posthoc['Comparison'] = duration_posthoc['Age Group 1'].astype(str) + ' vs ' + duration_posthoc['Age Group 2'].astype(str)

colors = ['green' if sig in ['*', '**', '***'] else 'gray' 
         for sig in duration_posthoc['Significance']]

ax.barh(duration_posthoc['Comparison'], duration_posthoc["Cohen's d"], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
ax.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5)
ax.axvline(x=-0.8, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel("Cohen's d", fontweight='bold')
ax.set_title('Post-hoc Effect Sizes: Sleep Duration', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Significance Matrix (Heatmap)
ax = axes[1, 1]

# Create significance matrix for sleep duration
sig_matrix = np.zeros((len(age_groups), len(age_groups)))
for _, row in duration_posthoc.iterrows():
    i = age_groups.index(row['Age Group 1'])
    j = age_groups.index(row['Age Group 2'])
    
    # Encode significance level
    if row['Significance'] == '***':
        val = 3
    elif row['Significance'] == '**':
        val = 2
    elif row['Significance'] == '*':
        val = 1
    else:
        val = 0
    
    sig_matrix[i, j] = val
    sig_matrix[j, i] = val

im = ax.imshow(sig_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
ax.set_xticks(range(len(age_groups)))
ax.set_yticks(range(len(age_groups)))
ax.set_xticklabels([f'{age}mo' for age in age_groups])
ax.set_yticklabels([f'{age}mo' for age in age_groups])
ax.set_title('Pairwise Significance: Sleep Duration', fontweight='bold')

# Add text annotations
for i in range(len(age_groups)):
    for j in range(len(age_groups)):
        if i != j:
            val = sig_matrix[i, j]
            text_color = 'white' if val > 1 else 'black'
            if val == 3:
                text = '***'
            elif val == 2:
                text = '**'
            elif val == 1:
                text = '*'
            else:
                text = 'ns'
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Significance Level')

plt.tight_layout()
plt.savefig(f'{output_folder}/statistical_fig1_hypothesis_tests.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: statistical_fig1_hypothesis_tests.png")
plt.close()

# ============================================================================
# VISUALIZATION: Means with Error Bars and Significance Stars
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (metric_name, col_name) in enumerate(list(metrics_to_test.items())[:6]):
    ax = axes[idx]
    
    # Calculate means and SDs
    means = []
    sds = []
    for age in age_groups:
        age_data = df[df['age_months'] == age][col_name].dropna()
        means.append(age_data.mean())
        sds.append(age_data.std())
    
    # Plot
    ax.errorbar(age_groups, means, yerr=sds, marker='o', markersize=10,
               linewidth=2, capsize=5, color='steelblue')
    
    # Add significance stars for most significant comparison
    metric_posthoc = posthoc_df[posthoc_df['Metric'] == metric_name]
    if len(metric_posthoc) > 0:
        # Find most significant
        most_sig = metric_posthoc.loc[metric_posthoc['p-value'].idxmin()]
        if most_sig['Significance'] != 'ns':
            age1_idx = age_groups.index(most_sig['Age Group 1'])
            age2_idx = age_groups.index(most_sig['Age Group 2'])
            
            y_max = max(means) + max(sds)
            y_star = y_max * 1.1
            
            ax.plot([age_groups[age1_idx], age_groups[age2_idx]], 
                   [y_star, y_star], 'k-', linewidth=1.5)
            ax.text((age_groups[age1_idx] + age_groups[age2_idx])/2, y_star,
                   most_sig['Significance'], ha='center', va='bottom', fontsize=14)
    
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel(metric_name, fontweight='bold')
    ax.set_title(metric_name, fontweight='bold')
    ax.set_xticks(age_groups)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/statistical_fig2_means_with_significance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: statistical_fig2_means_with_significance.png")
plt.close()

print("\n" + "="*70)
print("STATISTICAL TESTING COMPLETE")
print("="*70)
print(f"\nKey Findings:")
print(f"  • Significant age effects (p<0.05): {len(anova_df[anova_df['p-value'] < 0.05])}/{len(anova_df)} metrics")
print(f"  • Largest effect size: {anova_df.loc[anova_df['eta-squared'].idxmax(), 'Metric']} (η²={anova_df['eta-squared'].max():.3f})")
print(f"  • Total pairwise comparisons: {len(posthoc_df)}")
print(f"  • Significant comparisons (p<0.05, Bonferroni): {len(posthoc_df[posthoc_df['p-value (corrected)'] < 0.05])}")
print("\nFiles saved:")
print("  • anova_results.csv")
print("  • posthoc_comparisons.csv")
print("  • descriptive_statistics_by_age.csv")
print("  • statistical_fig1_hypothesis_tests.png")
print("  • statistical_fig2_means_with_significance.png")
print("="*70)