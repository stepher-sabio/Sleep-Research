"""
social_jetlag_analysis.py
Comprehensive weekday/weekend and social jet lag analysis

Research Questions:
1. Do sleep patterns differ between weekdays vs weekends?
2. What is the magnitude of social jet lag in toddlers?
3. Does Monday show recovery patterns after weekends?
4. Does social jet lag vary by age group?

Social Jet Lag (SJL) = |Weekend sleep midpoint - Weekday sleep midpoint|
Sleep Midpoint = (Bedtime + Wake time) / 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data_csv'
output_folder = '/Users/stepher/Desktop/Actigraphy2/visualizations/age_groups'
os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("SOCIAL JET LAG ANALYSIS")
print("="*70)
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_age(subject_id):
    """Extract age in months from subject ID"""
    try:
        parts = subject_id.split('_')
        age_part = [p for p in parts if 'mos' in p.lower()]
        if age_part:
            return int(age_part[0].replace('mos', '').replace('mo', ''))
        return None
    except:
        return None

def calculate_midpoint(bedtime_hour, waketime_hour):
    """Calculate sleep midpoint in 24-hour format"""
    # Handle overnight sleep
    if waketime_hour < bedtime_hour:
        # Wake time is next day
        waketime_hour += 24
    
    midpoint = (bedtime_hour + waketime_hour) / 2
    
    # Normalize to 0-24 range
    if midpoint >= 24:
        midpoint -= 24
    
    return midpoint

# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

print("Loading and processing data...")

all_subject_data = {}
subject_summary = []

for file_path in Path(input_folder).glob('*.csv'):
    subject_id = file_path.stem
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        sleep_df = df[df['interval_type'] == 'SLEEP'].copy()
        
        if len(sleep_df) == 0:
            continue
        
        # Clean numeric columns
        for col in ['sleep_time', 'efficiency', 'fragmentation']:
            if col in sleep_df.columns:
                sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce')
        
        sleep_df = sleep_df.dropna(subset=['sleep_time', 'start_date', 'start_time'])
        
        if len(sleep_df) == 0:
            continue
        
        # Parse dates and times
        sleep_df['date'] = pd.to_datetime(sleep_df['start_date'])
        sleep_df['bedtime'] = pd.to_datetime(sleep_df['start_date'] + ' ' + sleep_df['start_time'])
        sleep_df['waketime'] = pd.to_datetime(sleep_df['end_date'] + ' ' + sleep_df['end_time'])
        
        sleep_df['bedtime_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
        sleep_df['waketime_hour'] = sleep_df['waketime'].dt.hour + sleep_df['waketime'].dt.minute / 60
        
        # Day of week
        sleep_df['day_of_week'] = sleep_df['date'].dt.day_name()
        sleep_df['day_of_week_num'] = sleep_df['date'].dt.dayofweek  # 0=Mon, 6=Sun
        
        # Classify weekday vs weekend
        sleep_df['is_weekend'] = sleep_df['day_of_week_num'].isin([5, 6])  # Sat, Sun
        sleep_df['day_type'] = sleep_df['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        
        # Check if sufficient data (≥3 weekdays + ≥1 weekend)
        n_weekdays = sleep_df[~sleep_df['is_weekend']]['date'].nunique()
        n_weekends = sleep_df[sleep_df['is_weekend']]['date'].nunique()
        
        if n_weekdays < 3 or n_weekends < 1:
            continue
        
        # Store
        all_subject_data[subject_id] = sleep_df
        
    except Exception as e:
        print(f"  ⚠ Error loading {subject_id}: {e}")
        continue

print(f"Loaded {len(all_subject_data)} subjects with sufficient data\n")

# ============================================================================
# CALCULATE WEEKDAY/WEEKEND METRICS PER SUBJECT
# ============================================================================

print("Calculating weekday/weekend metrics...")

weekday_weekend_data = []

for subject_id, sleep_df in all_subject_data.items():
    age = extract_age(subject_id)
    
    # Aggregate by date to get daily totals
    daily = sleep_df.groupby(['date', 'is_weekend', 'day_type', 'day_of_week', 'day_of_week_num']).agg({
        'sleep_time': 'sum',
        'efficiency': 'mean',
        'fragmentation': 'mean',
        'bedtime_hour': 'mean',
        'waketime_hour': 'mean',
        'interval_number': 'count'  # Number of sleep periods (naps + nighttime)
    }).reset_index()
    
    daily.rename(columns={'interval_number': 'n_sleep_periods'}, inplace=True)
    
    # Separate weekday vs weekend
    weekday_data = daily[~daily['is_weekend']]
    weekend_data = daily[daily['is_weekend']]
    
    # Calculate midpoints for each day
    weekday_data = weekday_data.copy()
    weekend_data = weekend_data.copy()
    
    weekday_data['sleep_midpoint'] = weekday_data.apply(
        lambda row: calculate_midpoint(row['bedtime_hour'], row['waketime_hour']), axis=1
    )
    weekend_data['sleep_midpoint'] = weekend_data.apply(
        lambda row: calculate_midpoint(row['bedtime_hour'], row['waketime_hour']), axis=1
    )
    
    # Weekday averages (Mon-Fri)
    weekday_avg = {
        'bedtime': weekday_data['bedtime_hour'].mean(),
        'waketime': weekday_data['waketime_hour'].mean(),
        'duration': weekday_data['sleep_time'].mean(),
        'efficiency': weekday_data['efficiency'].mean(),
        'fragmentation': weekday_data['fragmentation'].mean(),
        'n_periods': weekday_data['n_sleep_periods'].mean(),
        'midpoint': weekday_data['sleep_midpoint'].mean(),
        'bedtime_sd': weekday_data['bedtime_hour'].std(),
        'waketime_sd': weekday_data['waketime_hour'].std()
    }
    
    # Weekend averages (Sat-Sun)
    weekend_avg = {
        'bedtime': weekend_data['bedtime_hour'].mean(),
        'waketime': weekend_data['waketime_hour'].mean(),
        'duration': weekend_data['sleep_time'].mean(),
        'efficiency': weekend_data['efficiency'].mean(),
        'fragmentation': weekend_data['fragmentation'].mean(),
        'n_periods': weekend_data['n_sleep_periods'].mean(),
        'midpoint': weekend_data['sleep_midpoint'].mean(),
        'bedtime_sd': weekend_data['bedtime_hour'].std(),
        'waketime_sd': weekend_data['waketime_hour'].std()
    }
    
    # Social Jet Lag (absolute difference in sleep midpoints)
    sjl_hours = abs(weekend_avg['midpoint'] - weekday_avg['midpoint'])
    sjl_minutes = sjl_hours * 60
    
    # Monday-specific analysis
    monday_data = daily[daily['day_of_week'] == 'Monday']
    other_weekdays = daily[(~daily['is_weekend']) & (daily['day_of_week'] != 'Monday')]
    
    if len(monday_data) > 0 and len(other_weekdays) > 0:
        monday_metrics = {
            'bedtime': monday_data['bedtime_hour'].mean(),
            'waketime': monday_data['waketime_hour'].mean(),
            'duration': monday_data['sleep_time'].mean(),
            'n_periods': monday_data['n_sleep_periods'].mean()
        }
        
        other_weekday_metrics = {
            'bedtime': other_weekdays['bedtime_hour'].mean(),
            'waketime': other_weekdays['waketime_hour'].mean(),
            'duration': other_weekdays['sleep_time'].mean(),
            'n_periods': other_weekdays['n_sleep_periods'].mean()
        }
    else:
        monday_metrics = {k: None for k in ['bedtime', 'waketime', 'duration', 'n_periods']}
        other_weekday_metrics = {k: None for k in ['bedtime', 'waketime', 'duration', 'n_periods']}
    
    weekday_weekend_data.append({
        'subject_id': subject_id,
        'age_months': age,
        
        # Weekday metrics
        'weekday_bedtime': weekday_avg['bedtime'],
        'weekday_waketime': weekday_avg['waketime'],
        'weekday_duration': weekday_avg['duration'],
        'weekday_efficiency': weekday_avg['efficiency'],
        'weekday_fragmentation': weekday_avg['fragmentation'],
        'weekday_n_periods': weekday_avg['n_periods'],
        'weekday_midpoint': weekday_avg['midpoint'],
        'weekday_bedtime_sd': weekday_avg['bedtime_sd'],
        
        # Weekend metrics
        'weekend_bedtime': weekend_avg['bedtime'],
        'weekend_waketime': weekend_avg['waketime'],
        'weekend_duration': weekend_avg['duration'],
        'weekend_efficiency': weekend_avg['efficiency'],
        'weekend_fragmentation': weekend_avg['fragmentation'],
        'weekend_n_periods': weekend_avg['n_periods'],
        'weekend_midpoint': weekend_avg['midpoint'],
        'weekend_bedtime_sd': weekend_avg['bedtime_sd'],
        
        # Differences
        'bedtime_diff': weekend_avg['bedtime'] - weekday_avg['bedtime'],
        'waketime_diff': weekend_avg['waketime'] - weekday_avg['waketime'],
        'duration_diff': weekend_avg['duration'] - weekday_avg['duration'],
        'efficiency_diff': weekend_avg['efficiency'] - weekday_avg['efficiency'],
        
        # Social Jet Lag
        'sjl_hours': sjl_hours,
        'sjl_minutes': sjl_minutes,
        
        # Monday recovery
        'monday_bedtime': monday_metrics['bedtime'],
        'monday_duration': monday_metrics['duration'],
        'monday_vs_other_weekday_bedtime_diff': (monday_metrics['bedtime'] - other_weekday_metrics['bedtime']) 
                                                 if monday_metrics['bedtime'] else None,
        'monday_vs_other_weekday_duration_diff': (monday_metrics['duration'] - other_weekday_metrics['duration'])
                                                  if monday_metrics['duration'] else None,
        
        # Sample sizes
        'n_weekdays': len(weekday_data),
        'n_weekends': len(weekend_data),
        'n_mondays': len(monday_data)
    })

sjl_df = pd.DataFrame(weekday_weekend_data)
sjl_df = sjl_df[sjl_df['age_months'].notna()]

print(f"Calculated metrics for {len(sjl_df)} subjects\n")

# Save
sjl_df.to_csv(f'{output_folder}/social_jetlag_by_subject.csv', index=False)
print(f"✓ Saved: social_jetlag_by_subject.csv\n")

# ============================================================================
# DAY-OF-WEEK PATTERNS
# ============================================================================

print("Analyzing day-of-week patterns...")

dow_patterns = []

for subject_id, sleep_df in all_subject_data.items():
    age = extract_age(subject_id)
    
    # Aggregate by date
    daily = sleep_df.groupby(['date', 'day_of_week', 'day_of_week_num']).agg({
        'sleep_time': 'sum',
        'efficiency': 'mean',
        'bedtime_hour': 'mean',
        'waketime_hour': 'mean',
        'interval_number': 'count'
    }).reset_index()
    
    # Average by day of week
    dow_avg = daily.groupby('day_of_week').agg({
        'sleep_time': 'mean',
        'bedtime_hour': 'mean',
        'waketime_hour': 'mean',
        'efficiency': 'mean',
        'interval_number': 'mean'
    }).reset_index()
    
    # Add metadata
    dow_avg['subject_id'] = subject_id
    dow_avg['age_months'] = age
    
    dow_patterns.append(dow_avg)

dow_df = pd.concat(dow_patterns, ignore_index=True)
dow_df = dow_df[dow_df['age_months'].notna()]

# Save
dow_df.to_csv(f'{output_folder}/day_of_week_patterns.csv', index=False)
print(f"✓ Saved: day_of_week_patterns.csv\n")

# ============================================================================
# STATISTICAL TESTING
# ============================================================================

print("Performing statistical tests...\n")

print("="*70)
print("PAIRED T-TESTS: Weekday vs Weekend")
print("="*70)

tests = {
    'Bedtime': ('weekday_bedtime', 'weekend_bedtime'),
    'Wake Time': ('weekday_waketime', 'weekend_waketime'),
    'Duration': ('weekday_duration', 'weekend_duration'),
    'Efficiency': ('weekday_efficiency', 'weekend_efficiency'),
    'Fragmentation': ('weekday_fragmentation', 'weekend_fragmentation'),
    'Sleep Periods': ('weekday_n_periods', 'weekend_n_periods')
}

test_results = []

for metric_name, (weekday_col, weekend_col) in tests.items():
    weekday_vals = sjl_df[weekday_col].dropna()
    weekend_vals = sjl_df[weekend_col].dropna()
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(weekday_vals, weekend_vals)
    
    # Effect size (Cohen's d for paired samples)
    diffs = weekend_vals - weekday_vals
    d = diffs.mean() / diffs.std()
    
    # Mean difference
    mean_diff = weekend_vals.mean() - weekday_vals.mean()
    
    # Significance
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    test_results.append({
        'Metric': metric_name,
        'Weekday_Mean': weekday_vals.mean(),
        'Weekend_Mean': weekend_vals.mean(),
        'Difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'Cohens_d': d,
        'Significance': sig
    })
    
    # Convert to appropriate units for display
    if 'time' in weekday_col.lower():
        unit = 'hours'
        display_diff = mean_diff
    elif 'duration' in weekday_col.lower():
        unit = 'minutes'
        display_diff = mean_diff
    elif 'periods' in weekday_col.lower():
        unit = 'periods'
        display_diff = mean_diff
    else:
        unit = ''
        display_diff = mean_diff
    
    print(f"\n{metric_name}:")
    print(f"  Weekday: {weekday_vals.mean():.2f} {unit}")
    print(f"  Weekend: {weekend_vals.mean():.2f} {unit}")
    print(f"  Difference: {display_diff:+.2f} {unit}")
    print(f"  t({len(weekday_vals)-1}) = {t_stat:.3f}, p = {p_value:.4f} {sig}")
    print(f"  Cohen's d = {d:.3f}")

test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv(f'{output_folder}/weekday_weekend_comparison.csv', index=False)
print(f"\n✓ Saved: weekday_weekend_comparison.csv\n")

# ============================================================================
# MONDAY RECOVERY ANALYSIS
# ============================================================================

print("="*70)
print("MONDAY RECOVERY ANALYSIS")
print("="*70)

monday_recovery_results = []

for age_group in sorted(sjl_df['age_months'].unique()):
    age_data = sjl_df[sjl_df['age_months'] == age_group]
    
    monday_bed = age_data['monday_bedtime'].dropna()
    monday_dur = age_data['monday_duration'].dropna()
    monday_vs_weekday_bed = age_data['monday_vs_other_weekday_bedtime_diff'].dropna()
    monday_vs_weekday_dur = age_data['monday_vs_other_weekday_duration_diff'].dropna()
    
    # Test if Monday differs from other weekdays
    if len(monday_vs_weekday_bed) > 3:
        t_bed, p_bed = stats.ttest_1samp(monday_vs_weekday_bed, 0)
    else:
        t_bed, p_bed = None, None
    
    if len(monday_vs_weekday_dur) > 3:
        t_dur, p_dur = stats.ttest_1samp(monday_vs_weekday_dur, 0)
    else:
        t_dur, p_dur = None, None
    
    monday_recovery_results.append({
        'age_months': age_group,
        'n_subjects': len(monday_bed),
        'monday_bedtime_mean': monday_bed.mean(),
        'monday_bedtime_sd': monday_bed.std(),
        'monday_duration_mean': monday_dur.mean(),
        'monday_duration_sd': monday_dur.std(),
        'monday_vs_other_bedtime_diff': monday_vs_weekday_bed.mean(),
        'monday_vs_other_duration_diff': monday_vs_weekday_dur.mean(),
        't_bedtime': t_bed,
        'p_bedtime': p_bed,
        't_duration': t_dur,
        'p_duration': p_dur
    })
    
    print(f"\nAge {age_group} months (n={len(monday_bed)}):")
    print(f"  Monday bedtime: {monday_bed.mean():.2f} hours")
    print(f"  Monday vs other weekdays: {monday_vs_weekday_bed.mean():+.2f} hours")
    if p_bed is not None:
        sig_bed = '***' if p_bed < 0.001 else '**' if p_bed < 0.01 else '*' if p_bed < 0.05 else 'ns'
        print(f"  t = {t_bed:.3f}, p = {p_bed:.4f} {sig_bed}")

monday_recovery_df = pd.DataFrame(monday_recovery_results)
monday_recovery_df.to_csv(f'{output_folder}/monday_recovery_analysis.csv', index=False)
print(f"\n✓ Saved: monday_recovery_analysis.csv\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...\n")

age_groups = sorted(sjl_df['age_months'].unique())
colors_age = {16: '#A23B72', 21: '#F18F01', 26: '#2E86AB', 31: '#C73E1D'}

# ============================================================================
# FIGURE 1: Weekday vs Weekend Patterns
# ============================================================================

print("Creating Figure 1: Weekday vs Weekend Patterns...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: Bedtime by Day of Week
ax = axes[0, 0]

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_bedtime_means = []
dow_bedtime_sems = []

for day in days_order:
    day_data = dow_df[dow_df['day_of_week'] == day]['bedtime_hour']
    dow_bedtime_means.append(day_data.mean())
    dow_bedtime_sems.append(day_data.sem())

x_pos = np.arange(len(days_order))
colors = ['steelblue']*5 + ['coral']*2

ax.bar(x_pos, dow_bedtime_means, yerr=dow_bedtime_sems, color=colors, 
      alpha=0.7, capsize=5, edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(days_order, rotation=45, ha='right')
ax.set_ylabel('Bedtime (hours)', fontweight='bold')
ax.set_title('Bedtime by Day of Week', fontweight='bold')
ax.axhline(y=np.mean(dow_bedtime_means[:5]), color='steelblue', 
          linestyle='--', alpha=0.5, linewidth=2, label='Weekday mean')
ax.axhline(y=np.mean(dow_bedtime_means[5:]), color='coral',
          linestyle='--', alpha=0.5, linewidth=2, label='Weekend mean')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Format y-axis
y_ticks = [18, 19, 20, 21, 22, 23]
y_labels = ['6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Panel B: Wake Time by Day of Week
ax = axes[0, 1]

dow_wake_means = []
dow_wake_sems = []

for day in days_order:
    day_data = dow_df[dow_df['day_of_week'] == day]['waketime_hour']
    dow_wake_means.append(day_data.mean())
    dow_wake_sems.append(day_data.sem())

ax.bar(x_pos, dow_wake_means, yerr=dow_wake_sems, color=colors,
      alpha=0.7, capsize=5, edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(days_order, rotation=45, ha='right')
ax.set_ylabel('Wake Time (hours)', fontweight='bold')
ax.set_title('Wake Time by Day of Week', fontweight='bold')
ax.axhline(y=np.mean(dow_wake_means[:5]), color='steelblue',
          linestyle='--', alpha=0.5, linewidth=2)
ax.axhline(y=np.mean(dow_wake_means[5:]), color='coral',
          linestyle='--', alpha=0.5, linewidth=2)
ax.grid(True, alpha=0.3, axis='y')

y_ticks = [6, 7, 8, 9, 10]
y_labels = ['6 AM', '7 AM', '8 AM', '9 AM', '10 AM']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Panel C: Duration by Day of Week
ax = axes[0, 2]

dow_dur_means = []
dow_dur_sems = []

for day in days_order:
    day_data = dow_df[dow_df['day_of_week'] == day]['sleep_time']
    dow_dur_means.append(day_data.mean() / 60)  # Convert to hours
    dow_dur_sems.append(day_data.sem() / 60)

ax.bar(x_pos, dow_dur_means, yerr=dow_dur_sems, color=colors,
      alpha=0.7, capsize=5, edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(days_order, rotation=45, ha='right')
ax.set_ylabel('Sleep Duration (hours)', fontweight='bold')
ax.set_title('Sleep Duration by Day of Week', fontweight='bold')
ax.axhline(y=np.mean(dow_dur_means[:5]), color='steelblue',
          linestyle='--', alpha=0.5, linewidth=2)
ax.axhline(y=np.mean(dow_dur_means[5:]), color='coral',
          linestyle='--', alpha=0.5, linewidth=2)
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Weekday vs Weekend Comparison (Bedtime)
ax = axes[1, 0]

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    x_jitter = np.random.normal(age, 0.2, size=len(age_data))
    ax.scatter(age_data['weekday_bedtime'], age_data['weekend_bedtime'],
              s=80, alpha=0.6, color=colors_age.get(age, 'gray'),
              edgecolors='white', linewidth=1.5)

# Diagonal line
ax.plot([18, 24], [18, 24], 'k--', linewidth=2, alpha=0.5, label='No difference')
ax.set_xlabel('Weekday Bedtime (hours)', fontweight='bold')
ax.set_ylabel('Weekend Bedtime (hours)', fontweight='bold')
ax.set_title('Weekday vs Weekend Bedtime', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel E: Weekday vs Weekend Comparison (Duration)
ax = axes[1, 1]

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    ax.scatter(age_data['weekday_duration'] / 60, age_data['weekend_duration'] / 60,
              s=80, alpha=0.6, color=colors_age.get(age, 'gray'),
              label=f'{age} months', edgecolors='white', linewidth=1.5)

ax.plot([8, 14], [8, 14], 'k--', linewidth=2, alpha=0.5)
ax.set_xlabel('Weekday Duration (hours)', fontweight='bold')
ax.set_ylabel('Weekend Duration (hours)', fontweight='bold')
ax.set_title('Weekday vs Weekend Duration', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel F: Number of Sleep Periods
ax = axes[1, 2]

dow_periods_means = []
dow_periods_sems = []

for day in days_order:
    day_data = dow_df[dow_df['day_of_week'] == day]['interval_number']
    dow_periods_means.append(day_data.mean())
    dow_periods_sems.append(day_data.sem())

ax.bar(x_pos, dow_periods_means, yerr=dow_periods_sems, color=colors,
      alpha=0.7, capsize=5, edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(days_order, rotation=45, ha='right')
ax.set_ylabel('Number of Sleep Periods', fontweight='bold')
ax.set_title('Sleep Consolidation by Day', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_folder}/social_jetlag_fig1_weekday_weekend.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: social_jetlag_fig1_weekday_weekend.png")
plt.close()

# ============================================================================
# FIGURE 2: Social Jet Lag Analysis
# ============================================================================

print("Creating Figure 2: Social Jet Lag Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Sleep Midpoint - Weekday vs Weekend
ax = axes[0, 0]

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    ax.scatter(age_data['weekday_midpoint'], age_data['weekend_midpoint'],
              s=100, alpha=0.6, color=colors_age.get(age, 'gray'),
              label=f'{age} months', edgecolors='white', linewidth=1.5)

ax.plot([0, 8], [0, 8], 'k--', linewidth=2, alpha=0.5, label='No SJL')
ax.set_xlabel('Weekday Sleep Midpoint (hours)', fontweight='bold')
ax.set_ylabel('Weekend Sleep Midpoint (hours)', fontweight='bold')
ax.set_title('Sleep Midpoint: Weekday vs Weekend', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Format axes
x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
x_labels = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_yticks(x_ticks)
ax.set_yticklabels(x_labels)

# Panel B: SJL Distribution
ax = axes[0, 1]

ax.hist(sjl_df['sjl_minutes'], bins=20, color='steelblue', alpha=0.7, edgecolor='white', linewidth=1.5)
ax.axvline(x=sjl_df['sjl_minutes'].mean(), color='red', linestyle='--', linewidth=2,
          label=f"Mean = {sjl_df['sjl_minutes'].mean():.1f} min")
ax.axvline(x=sjl_df['sjl_minutes'].median(), color='orange', linestyle='--', linewidth=2,
          label=f"Median = {sjl_df['sjl_minutes'].median():.1f} min")
ax.set_xlabel('Social Jet Lag (minutes)', fontweight='bold')
ax.set_ylabel('Number of Subjects', fontweight='bold')
ax.set_title('Social Jet Lag Distribution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel C: SJL by Age Group
ax = axes[1, 0]

sjl_means = sjl_df.groupby('age_months')['sjl_minutes'].mean()
sjl_stds = sjl_df.groupby('age_months')['sjl_minutes'].std()

# Plot individual points
for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    x_jitter = np.random.normal(age, 0.3, size=len(age_data))
    ax.scatter(x_jitter, age_data['sjl_minutes'],
              s=60, alpha=0.4, color=colors_age.get(age, 'gray'))

# Overlay means
ax.errorbar(age_groups, sjl_means, yerr=sjl_stds,
           marker='o', markersize=12, linewidth=3, capsize=5,
           color='black', label='Mean ± SD', zorder=10)

ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Social Jet Lag (minutes)', fontweight='bold')
ax.set_title('Social Jet Lag by Age Group', fontweight='bold')
ax.set_xticks(age_groups)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel D: SJL vs Sleep Quality (Efficiency)
ax = axes[1, 1]

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    ax.scatter(age_data['sjl_minutes'], age_data['weekday_efficiency'],
              s=80, alpha=0.6, color=colors_age.get(age, 'gray'),
              label=f'{age} months', edgecolors='white', linewidth=1.5)

# Add trendline
x = sjl_df['sjl_minutes'].dropna()
y = sjl_df['weekday_efficiency'].dropna()
if len(x) > 5:
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5)
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Social Jet Lag (minutes)', fontweight='bold')
ax.set_ylabel('Weekday Sleep Efficiency (%)', fontweight='bold')
ax.set_title('SJL vs Sleep Quality', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/social_jetlag_fig2_sjl_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: social_jetlag_fig2_sjl_analysis.png")
plt.close()

# ============================================================================
# FIGURE 3: Monday Recovery Effects
# ============================================================================

print("Creating Figure 3: Monday Recovery Effects...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Monday vs Other Weekdays (Bedtime)
ax = axes[0, 0]

monday_bed_data = []
other_weekday_bed_data = []
labels = []

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    monday_bed_data.append(age_data['monday_bedtime'].dropna().values)
    
    # Calculate other weekday average
    other_avg = []
    for _, row in age_data.iterrows():
        if pd.notna(row['monday_bedtime']) and pd.notna(row['weekday_bedtime']):
            # Approximate other weekdays
            other_weekday_avg = row['weekday_bedtime']
            other_avg.append(other_weekday_avg)
    other_weekday_bed_data.append(np.array(other_avg))
    labels.append(f'{age}mo')

x_pos = np.arange(len(age_groups))
width = 0.35

monday_means = [np.mean(d) if len(d) > 0 else 0 for d in monday_bed_data]
monday_sems = [np.std(d)/np.sqrt(len(d)) if len(d) > 0 else 0 for d in monday_bed_data]
other_means = [np.mean(d) if len(d) > 0 else 0 for d in other_weekday_bed_data]
other_sems = [np.std(d)/np.sqrt(len(d)) if len(d) > 0 else 0 for d in other_weekday_bed_data]

ax.bar(x_pos - width/2, monday_means, width, yerr=monday_sems,
      label='Monday', color='coral', alpha=0.8, capsize=5)
ax.bar(x_pos + width/2, other_means, width, yerr=other_sems,
      label='Tue-Fri', color='steelblue', alpha=0.8, capsize=5)

ax.set_xlabel('Age (months)', fontweight='bold')
ax.set_ylabel('Bedtime (hours)', fontweight='bold')
ax.set_title('Monday vs Other Weekdays: Bedtime', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Weekend Sleep Duration vs Monday Recovery
ax = axes[0, 1]

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    x = age_data['weekend_duration'] - age_data['weekday_duration']  # Weekend catch-up
    y = age_data['monday_vs_other_weekday_duration_diff'].dropna()
    
    if len(y) > 0:
        ax.scatter(x[:len(y)], y,
                  s=80, alpha=0.6, color=colors_age.get(age, 'gray'),
                  label=f'{age} months', edgecolors='white', linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Weekend Extra Sleep (minutes)', fontweight='bold')
ax.set_ylabel('Monday Sleep Change (minutes)', fontweight='bold')
ax.set_title('Weekend Catch-up vs Monday Recovery', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: Day-to-Day Trajectory (Sun → Mon → Tue)
ax = axes[1, 0]

# Aggregate data across all subjects for visualization
trajectory_data = []

for subject_id, sleep_df in all_subject_data.items():
    daily = sleep_df.groupby(['date', 'day_of_week']).agg({
        'sleep_time': 'sum',
        'bedtime_hour': 'mean'
    }).reset_index()
    
    # Find sequences of Sun → Mon → Tue
    for i in range(len(daily) - 2):
        if (daily.iloc[i]['day_of_week'] == 'Sunday' and
            daily.iloc[i+1]['day_of_week'] == 'Monday' and
            daily.iloc[i+2]['day_of_week'] == 'Tuesday'):
            
            trajectory_data.append({
                'sunday_bedtime': daily.iloc[i]['bedtime_hour'],
                'monday_bedtime': daily.iloc[i+1]['bedtime_hour'],
                'tuesday_bedtime': daily.iloc[i+2]['bedtime_hour']
            })

if len(trajectory_data) > 0:
    traj_df = pd.DataFrame(trajectory_data)
    
    days = ['Sunday', 'Monday', 'Tuesday']
    means = [traj_df['sunday_bedtime'].mean(), 
            traj_df['monday_bedtime'].mean(),
            traj_df['tuesday_bedtime'].mean()]
    sems = [traj_df['sunday_bedtime'].sem(),
           traj_df['monday_bedtime'].sem(),
           traj_df['tuesday_bedtime'].sem()]
    
    ax.plot(days, means, marker='o', markersize=10, linewidth=3, color='steelblue')
    ax.errorbar(days, means, yerr=sems, fmt='none', capsize=5, color='steelblue')
    
    ax.set_ylabel('Bedtime (hours)', fontweight='bold')
    ax.set_title('Weekend → Monday Transition', fontweight='bold')
    ax.grid(True, alpha=0.3)

# Panel D: Recovery Summary by Age
ax = axes[1, 1]

recovery_metric = []
recovery_age = []

for age in age_groups:
    age_data = sjl_df[sjl_df['age_months'] == age]
    recovery = age_data['monday_vs_other_weekday_bedtime_diff'].dropna()
    
    if len(recovery) > 0:
        recovery_metric.extend(recovery.values)
        recovery_age.extend([age] * len(recovery))

if len(recovery_metric) > 0:
    data_list = [sjl_df[sjl_df['age_months'] == age]['monday_vs_other_weekday_bedtime_diff'].dropna()
                for age in age_groups]
    
    bp = ax.boxplot(data_list, labels=age_groups, patch_artist=True, widths=0.6)
    
    for patch, age in zip(bp['boxes'], age_groups):
        patch.set_facecolor(colors_age.get(age, 'gray'))
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Age (months)', fontweight='bold')
    ax.set_ylabel('Monday - Other Weekdays (hours)', fontweight='bold')
    ax.set_title('Monday Recovery Effect by Age', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_folder}/social_jetlag_fig3_monday_recovery.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: social_jetlag_fig3_monday_recovery.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*70)
print("SOCIAL JET LAG ANALYSIS COMPLETE")
print("="*70)

print("\nKey Findings:")
print(f"\n1. SOCIAL JET LAG:")
print(f"   • Mean SJL: {sjl_df['sjl_minutes'].mean():.1f} ± {sjl_df['sjl_minutes'].std():.1f} minutes")
print(f"   • Median SJL: {sjl_df['sjl_minutes'].median():.1f} minutes")
print(f"   • Range: {sjl_df['sjl_minutes'].min():.1f} - {sjl_df['sjl_minutes'].max():.1f} minutes")

print(f"\n2. WEEKDAY vs WEEKEND:")
bedtime_diff = sjl_df['bedtime_diff'].mean() * 60
wake_diff = sjl_df['waketime_diff'].mean() * 60
dur_diff = sjl_df['duration_diff'].mean()
print(f"   • Bedtime shift: {bedtime_diff:+.1f} minutes")
print(f"   • Wake time shift: {wake_diff:+.1f} minutes")
print(f"   • Duration change: {dur_diff:+.1f} minutes")

print(f"\n3. MONDAY RECOVERY:")
monday_data = sjl_df['monday_vs_other_weekday_bedtime_diff'].dropna()
if len(monday_data) > 0:
    print(f"   • Monday bedtime vs other weekdays: {monday_data.mean()*60:+.1f} minutes")
    print(f"   • {len(monday_data[monday_data > 0])} subjects sleep earlier on Monday")
    print(f"   • {len(monday_data[monday_data < 0])} subjects sleep later on Monday")

print(f"\n4. AGE EFFECTS:")
for age in age_groups:
    age_sjl = sjl_df[sjl_df['age_months'] == age]['sjl_minutes']
    print(f"   • {age} months: SJL = {age_sjl.mean():.1f} ± {age_sjl.std():.1f} min (n={len(age_sjl)})")

print("\nFiles saved:")
print("  • social_jetlag_by_subject.csv")
print("  • day_of_week_patterns.csv")
print("  • weekday_weekend_comparison.csv")
print("  • monday_recovery_analysis.csv")
print("  • social_jetlag_fig1_weekday_weekend.png")
print("  • social_jetlag_fig2_sjl_analysis.png")
print("  • social_jetlag_fig3_monday_recovery.png")

print("\n" + "="*70)