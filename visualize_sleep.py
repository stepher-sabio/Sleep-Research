"""
visualize_sleep.py
Sleep data visualization functions using matplotlib and seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# Set style for all plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def plot_sleep_raster(sleep_df, subject_id, save_path=None):
    """
    Create a sleep raster plot showing ALL sleep periods.
    Dynamic x-axis ensures all sleep is visible.
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with SLEEP intervals
    subject_id : str
        Subject identifier for title
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Prepare data
    if 'bedtime' not in sleep_df.columns:
        sleep_df = sleep_df.copy()
        sleep_df['bedtime'] = pd.to_datetime(
            sleep_df['start_date'] + ' ' + sleep_df['start_time']
        )
        sleep_df['waketime'] = pd.to_datetime(
            sleep_df['end_date'] + ' ' + sleep_df['end_time']
        )
    
    # Get unique dates
    dates = sleep_df['start_date'].unique()
    dates = sorted(dates)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, len(dates) * 0.4)))
    
    # Single color for all sleep
    color = '#2E86AB'  # Blue
    
    # ⭐ Track all bed and wake hours to determine x-axis range
    all_bed_hours = []
    all_wake_hours = []
    all_plot_positions = []  # Track actual plot positions
    
    # First pass: collect all hours
    for i, date in enumerate(dates):
        day_sleep = sleep_df[sleep_df['start_date'] == date]
        
        for _, row in day_sleep.iterrows():
            bed_hour = row['bedtime'].hour + row['bedtime'].minute / 60
            wake_hour = row['waketime'].hour + row['waketime'].minute / 60
            
            all_bed_hours.append(bed_hour)
            all_wake_hours.append(wake_hour)
            
            # Track where we'll actually plot
            if wake_hour < bed_hour:  # Overnight
                all_plot_positions.extend([bed_hour, 24, 24 + wake_hour])
            else:  # Same day
                all_plot_positions.extend([bed_hour, wake_hour])
    
    # Determine x-axis range
    # Find earliest bedtime and latest wake time
    earliest_bed = min(all_bed_hours)
    latest_wake = max(all_wake_hours)
    
    # Check if any sleep crosses midnight
    has_overnight = any(w < b for w, b in zip(all_wake_hours, all_bed_hours))
    
    if has_overnight:
        # Need to show across midnight
        # Start from earliest bedtime (usually evening)
        # Extend past midnight to show wake times
        x_min = max(0, earliest_bed - 1)
        x_max = 24 + max(all_wake_hours) + 1
    else:
        # All sleep is within same calendar day
        x_min = max(0, earliest_bed - 1)
        x_max = min(24, latest_wake + 1)
    
    # Plot each sleep period
    for i, date in enumerate(dates):
        day_sleep = sleep_df[sleep_df['start_date'] == date]
        
        for _, row in day_sleep.iterrows():
            # Convert times to hours for plotting
            bed_hour = row['bedtime'].hour + row['bedtime'].minute / 60
            wake_hour = row['waketime'].hour + row['waketime'].minute / 60
            
            # Handle overnight sleep (crosses midnight)
            if wake_hour < bed_hour:
                # Split into two bars to show continuity across midnight
                # Part 1: bedtime to midnight
                duration1 = 24 - bed_hour
                ax.barh(i, duration1, left=bed_hour, height=0.8,
                       color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Part 2: midnight to wake time (add 24 to plot on "next day" side)
                ax.barh(i, wake_hour, left=24, height=0.8,
                       color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
            else:
                # Daytime sleep: single bar
                duration = wake_hour - bed_hour
                ax.barh(i, duration, left=bed_hour, height=0.8,
                       color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Formatting
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates)
    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(f'Sleep Raster Plot - {subject_id}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # ⭐ Set x-axis limits based on actual data
    ax.set_xlim(x_min, x_max)
    
    # ⭐ Create dynamic tick labels
    # Determine appropriate tick spacing
    x_range = x_max - x_min
    if x_range <= 12:
        tick_interval = 2
    elif x_range <= 24:
        tick_interval = 3
    else:
        tick_interval = 3
    
    hour_ticks = []
    hour_labels = []
    
    current_tick = int(x_min)
    while current_tick <= x_max:
        hour_ticks.append(current_tick)
        
        # Determine label
        if current_tick >= 24:
            # Next day hours
            h = current_tick - 24
            if h == 0:
                hour_labels.append('12 AM')
            elif h < 12:
                hour_labels.append(f'{h} AM')
            elif h == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{h-12} PM')
        else:
            # Same day hours
            h = current_tick
            if h == 0:
                hour_labels.append('12 AM')
            elif h < 12:
                hour_labels.append(f'{int(h)} AM')
            elif h == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{int(h-12)} PM')
        
        current_tick += tick_interval
    
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels, rotation=0)
    
    # Add vertical line for midnight (if visible)
    if x_min < 24 < x_max:
        ax.axvline(x=24, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def analyze_naps_vs_nighttime(sleep_df, subject_id):
    """
    Separate analysis for naps vs nighttime sleep.
    Simplified to just two categories.
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with SLEEP intervals
    subject_id : str
        Subject identifier
    
    Returns:
    --------
    dict : Dictionary with separate statistics for naps and nighttime
    """
    
    # Prepare data
    if 'bedtime' not in sleep_df.columns:
        sleep_df = sleep_df.copy()
        sleep_df['bedtime'] = pd.to_datetime(
            sleep_df['start_date'] + ' ' + sleep_df['start_time']
        )
        sleep_df['waketime'] = pd.to_datetime(
            sleep_df['end_date'] + ' ' + sleep_df['end_time']
        )
    
    # ⭐ SIMPLIFIED categorization
    def categorize_sleep(row):
        duration = row['sleep_time']
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            return "Unknown"
        
        if pd.isna(duration):
            return "Unknown"
        
        # Simple: Nighttime (>= 180 min) or Nap (< 180 min)
        if duration >= 180:
            return "Nighttime Sleep"
        else:
            return "Nap"
    
    sleep_df['sleep_category'] = sleep_df.apply(categorize_sleep, axis=1)
    
    # Separate nighttime and naps
    nighttime = sleep_df[sleep_df['sleep_category'] == 'Nighttime Sleep']
    naps = sleep_df[sleep_df['sleep_category'] == 'Nap']
    
    results = {'subject_id': subject_id}
    
    # Nighttime statistics
    if len(nighttime) > 0:
        nighttime['bedtime_hour'] = nighttime['bedtime'].dt.hour + nighttime['bedtime'].dt.minute / 60
        nighttime['waketime_hour'] = nighttime['waketime'].dt.hour + nighttime['waketime'].dt.minute / 60
        
        results['nighttime'] = {
            'count': len(nighttime),
            'duration_mean': nighttime['sleep_time'].mean(),
            'duration_std': nighttime['sleep_time'].std(),
            'efficiency_mean': nighttime['efficiency'].mean(),
            'efficiency_std': nighttime['efficiency'].std(),
            'fragmentation_mean': nighttime['fragmentation'].mean(),
            'fragmentation_std': nighttime['fragmentation'].std(),
            'onset_latency_mean': nighttime['onset_latency'].mean(),
            'wake_time_mean': nighttime['wake_time'].mean(),
            'bedtime_mean': nighttime['bedtime_hour'].mean(),
            'waketime_mean': nighttime['waketime_hour'].mean()
        }
    else:
        results['nighttime'] = None
    
    # Nap statistics (simplified - no breakdown by type)
    if len(naps) > 0:
        naps['start_hour'] = naps['bedtime'].dt.hour + naps['bedtime'].dt.minute / 60
        naps['end_hour'] = naps['waketime'].dt.hour + naps['waketime'].dt.minute / 60
        
        # Get unique dates to calculate naps per day
        total_days = sleep_df['start_date'].nunique()
        
        results['naps'] = {
            'total_count': len(naps),
            'naps_per_day': len(naps) / total_days,
            'duration_mean': naps['sleep_time'].mean(),
            'duration_std': naps['sleep_time'].std(),
            'efficiency_mean': naps['efficiency'].mean(),
            'efficiency_std': naps['efficiency'].std(),
            'fragmentation_mean': naps['fragmentation'].mean(),
            'fragmentation_std': naps['fragmentation'].std(),
            'onset_latency_mean': naps['onset_latency'].mean(),
            'start_time_mean': naps['start_hour'].mean(),
            'start_time_std': naps['start_hour'].std(),
            'end_time_mean': naps['end_hour'].mean(),
            'end_time_std': naps['end_hour'].std()
        }
    else:
        results['naps'] = None
    
    return results


def print_nap_nighttime_summary(results):
    """
    Print formatted summary of nap vs nighttime analysis.
    Simplified version with just two categories.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_naps_vs_nighttime()
    """
    
    print("\n" + "="*70)
    print(f"NAP vs NIGHTTIME ANALYSIS - {results['subject_id']}")
    print("="*70)
    
    # Nighttime statistics
    if results['nighttime']:
        night = results['nighttime']
        print("\n--- NIGHTTIME SLEEP ---")
        print(f"  Count: {night['count']} nights")
        print(f"  Duration: {night['duration_mean']:.1f} ± {night['duration_std']:.1f} minutes ({night['duration_mean']/60:.1f} hours)")
        print(f"  Efficiency: {night['efficiency_mean']:.1f}% ± {night['efficiency_std']:.1f}%")
        print(f"  Fragmentation: {night['fragmentation_mean']:.2f} ± {night['fragmentation_std']:.2f}")
        print(f"  Onset Latency: {night['onset_latency_mean']:.1f} minutes")
        print(f"  WASO: {night['wake_time_mean']:.1f} minutes")
        print(f"  Bedtime: {format_time(night['bedtime_mean'])}")
        print(f"  Wake Time: {format_time(night['waketime_mean'])}")
    else:
        print("\n--- NIGHTTIME SLEEP ---")
        print("  No nighttime sleep data")
    
    # Nap statistics (simplified - no type breakdown)
    if results['naps']:
        nap = results['naps']
        print("\n--- NAPS ---")
        print(f"  Total Naps: {nap['total_count']}")
        print(f"  Naps per Day: {nap['naps_per_day']:.1f}")
        print(f"  Duration: {nap['duration_mean']:.1f} ± {nap['duration_std']:.1f} minutes")
        print(f"  Efficiency: {nap['efficiency_mean']:.1f}% ± {nap['efficiency_std']:.1f}%")
        print(f"  Fragmentation: {nap['fragmentation_mean']:.2f} ± {nap['fragmentation_std']:.2f}")
        print(f"  Onset Latency: {nap['onset_latency_mean']:.1f} minutes")
        print(f"  Onset Time: {format_time(nap['start_time_mean'])} ± {nap['start_time_std']*60:.0f} min")
        print(f"  End Time: {format_time(nap['end_time_mean'])} ± {nap['end_time_std']*60:.0f} min")
    else:
        print("\n--- NAPS ---")
        print("  No naps detected")
    
    print("="*70 + "\n")


def format_time(hour_decimal):
    """Convert decimal hour to readable time format."""
    if pd.isna(hour_decimal):
        return "N/A"
    
    hours = int(hour_decimal)
    minutes = int((hour_decimal - hours) * 60)
    
    if hours == 0:
        period = "AM"
        display_hour = 12
    elif hours < 12:
        period = "AM"
        display_hour = hours
    elif hours == 12:
        period = "PM"
        display_hour = 12
    else:
        period = "PM"
        display_hour = hours - 12
    
    return f"{display_hour}:{minutes:02d} {period}"


def plot_sleep_duration(daily_summary, subject_id, save_path=None):
    """
    Create time series plot of daily sleep duration.
    
    Parameters:
    -----------
    daily_summary : pandas.DataFrame
        DataFrame with daily aggregated data
    subject_id : str
        Subject identifier
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Convert dates
    dates = pd.to_datetime(daily_summary['start_date'])
    sleep_hours = daily_summary['sleep_time'] / 60
    
    # Plot - NO REFERENCE LINE
    ax.plot(dates, sleep_hours, marker='o', linewidth=2, markersize=8,
           color='#2E86AB', label='Total Sleep Time')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Sleep (hours)', fontsize=12, fontweight='bold')
    ax.set_title(f'Daily Sleep Duration - {subject_id}', fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on points
    for date, hours in zip(dates, sleep_hours):
        ax.annotate(f'{hours:.1f}h', 
                   xy=(date, hours), 
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   color='#2E86AB')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    y_min = max(0, min(sleep_hours) - 0.5)
    y_max = max(sleep_hours) + 0.5
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_sleep_quality(daily_summary, subject_id, save_path=None):
    """
    Create multi-line plot of sleep quality metrics over time.
    
    Parameters:
    -----------
    daily_summary : pandas.DataFrame
        DataFrame with daily aggregated data
    subject_id : str
        Subject identifier
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    dates = pd.to_datetime(daily_summary['start_date'])
    
    # Plot 1: Sleep Efficiency - NO THRESHOLD
    ax1 = axes[0]
    ax1.plot(dates, daily_summary['efficiency'], marker='o', linewidth=2,
            markersize=6, color='#2E86AB', label='Sleep Efficiency')
    ax1.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Sleep Quality Metrics - {subject_id}', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fragmentation - NO THRESHOLD
    ax2 = axes[1]
    ax2.plot(dates, daily_summary['fragmentation'], marker='s', linewidth=2,
            markersize=6, color='#F18F01', label='Fragmentation Index')
    ax2.set_ylabel('Fragmentation', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Onset Latency & WASO - NO THRESHOLD
    ax3 = axes[2]
    ax3.plot(dates, daily_summary['onset_latency'], marker='^', linewidth=2,
            markersize=6, color='#A23B72', label='Onset Latency')
    ax3.plot(dates, daily_summary['wake_time'], marker='v', linewidth=2,
            markersize=6, color='#C73E1D', label='WASO')
    ax3.set_ylabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis for bottom plot
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_timing_consistency(sleep_df, subject_id, save_path=None):
    """
    Create plot showing sleep timing categorized by time of day.
    Shows start and end times with means for each category.
    
    Categories:
    - Morning: 10 AM - 11:59 AM
    - Afternoon: 12 PM - 5:59 PM
    - Evening: 6 PM - 9:59 AM (everything else)
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with SLEEP intervals
    subject_id : str
        Subject identifier
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Prepare data
    if 'bedtime' not in sleep_df.columns:
        sleep_df = sleep_df.copy()
        sleep_df['bedtime'] = pd.to_datetime(
            sleep_df['start_date'] + ' ' + sleep_df['start_time']
        )
        sleep_df['waketime'] = pd.to_datetime(
            sleep_df['end_date'] + ' ' + sleep_df['end_time']
        )
    
    # Extract times as hours
    sleep_df['start_hour'] = sleep_df['bedtime'].dt.hour + sleep_df['bedtime'].dt.minute / 60
    sleep_df['end_hour'] = sleep_df['waketime'].dt.hour + sleep_df['waketime'].dt.minute / 60
    
    # ⭐ Categorize by START time - 3 categories only
    def categorize_by_time(start_hour):
        """Categorize sleep period by start time."""
        if 10 <= start_hour < 12:
            return "Morning"
        elif 12 <= start_hour < 18:
            return "Afternoon"
        else:
            # Everything else is Evening (6 PM - 9:59 AM)
            return "Evening"
    
    sleep_df['time_category'] = sleep_df['start_hour'].apply(categorize_by_time)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ⭐ Color map - 3 categories only
    colors = {
        'Morning': '#A23B72',      # Purple
        'Afternoon': '#F18F01',    # Orange
        'Evening': '#2E86AB'       # Blue
    }
    
    markers = {
        'Morning': 'o',
        'Afternoon': 's',
        'Evening': '^'
    }
    
    dates = pd.to_datetime(sleep_df['start_date'])
    
    # Plot each category
    for category in ['Morning', 'Afternoon', 'Evening']:
        cat_data = sleep_df[sleep_df['time_category'] == category]
        
        if len(cat_data) == 0:
            continue
        
        cat_dates = pd.to_datetime(cat_data['start_date'])
        
        # Plot start times (filled markers)
        ax.scatter(cat_dates, cat_data['start_hour'], 
                  s=100, alpha=0.7,
                  color=colors[category], 
                  marker=markers[category],
                  label=f'{category} (start)',
                  edgecolors='white', 
                  linewidth=1.5)
        
        # Plot end times (hollow markers)
        ax.scatter(cat_dates, cat_data['end_hour'], 
                  s=100, alpha=0.7,
                  color=colors[category], 
                  marker=markers[category],
                  facecolors='none',
                  edgecolors=colors[category],
                  linewidth=2,
                  label=f'{category} (end)')
        
        # ⭐ Calculate and plot mean lines (dashed)
        mean_start = cat_data['start_hour'].mean()
        mean_end = cat_data['end_hour'].mean()
        
        # Plot mean start time line
        ax.axhline(y=mean_start, 
                  color=colors[category], 
                  linestyle='--', 
                  alpha=0.5,
                  linewidth=2,
                  label=f'{category} mean start ({format_time(mean_start)})')
        
        # Plot mean end time line
        ax.axhline(y=mean_end, 
                  color=colors[category], 
                  linestyle=':', 
                  alpha=0.5,
                  linewidth=2,
                  label=f'{category} mean end ({format_time(mean_end)})')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (24-hour)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sleep Timing by Time of Day - {subject_id}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Y-axis: times
    y_ticks = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    y_labels = ['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(0, 24)
    
    # Add shaded regions for categories
    ax.axhspan(10, 12, alpha=0.05, color=colors['Morning'], zorder=0)
    ax.axhspan(12, 18, alpha=0.05, color=colors['Afternoon'], zorder=0)
    # Evening spans two regions (6 PM - midnight and midnight - 10 AM)
    ax.axhspan(18, 24, alpha=0.05, color=colors['Evening'], zorder=0)
    ax.axhspan(0, 10, alpha=0.05, color=colors['Evening'], zorder=0)
    
    # X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    # Legend (place outside to avoid clutter)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def create_comprehensive_dashboard(sleep_df, daily_summary, subject_id, save_path=None):
    """
    Create a comprehensive dashboard with multiple subplots.
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with SLEEP intervals
    daily_summary : pandas.DataFrame
        DataFrame with daily aggregated data
    subject_id : str
        Subject identifier
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Overall title
    fig.suptitle(f'Sleep Analysis Dashboard - {subject_id}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Subplot 1: Sleep Raster (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_sleep_raster_on_axis(ax1, sleep_df, subject_id)
    
    # Subplot 2: Sleep Duration (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_sleep_duration_on_axis(ax2, daily_summary)
    
    # Subplot 3: Sleep Efficiency (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_efficiency_on_axis(ax3, daily_summary)
    
    # Subplot 4: Fragmentation (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    plot_fragmentation_on_axis(ax4, daily_summary)
    
    # Subplot 5: Timing Consistency (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    plot_timing_on_axis(ax5, sleep_df)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Helper functions for dashboard (simplified versions)
def plot_sleep_raster_on_axis(ax, sleep_df, subject_id):
    """Simplified raster plot for dashboard."""
    # [Similar to plot_sleep_raster but plots on provided axis]
    # Implementation similar to above but without fig creation
    pass  # Implement if needed


def plot_sleep_duration_on_axis(ax, daily_summary):
    """Simplified duration plot for dashboard."""
    dates = pd.to_datetime(daily_summary['start_date'])
    sleep_hours = daily_summary['sleep_time'] / 60
    ax.plot(dates, sleep_hours, marker='o', linewidth=2, color='#2E86AB')
    ax.set_ylabel('Sleep (hours)', fontsize=10, fontweight='bold')
    ax.set_title('Daily Sleep Duration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_efficiency_on_axis(ax, daily_summary):
    """Simplified efficiency plot for dashboard."""
    dates = pd.to_datetime(daily_summary['start_date'])
    ax.plot(dates, daily_summary['efficiency'], marker='o', linewidth=2, color='#2E86AB')
    ax.axhline(y=85, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('Efficiency (%)', fontsize=10, fontweight='bold')
    ax.set_title('Sleep Efficiency', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_fragmentation_on_axis(ax, daily_summary):
    """Simplified fragmentation plot for dashboard."""
    dates = pd.to_datetime(daily_summary['start_date'])
    ax.plot(dates, daily_summary['fragmentation'], marker='s', linewidth=2, color='#F18F01')
    ax.set_ylabel('Fragmentation', fontsize=10, fontweight='bold')
    ax.set_title('Sleep Fragmentation', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_timing_on_axis(ax, sleep_df):
    """Simplified timing plot for dashboard."""
    # Similar to plot_timing_consistency but on provided axis
    pass  # Implement if needed