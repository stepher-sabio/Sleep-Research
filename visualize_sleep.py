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
    Create a sleep raster plot (actogram) showing NIGHTTIME SLEEP ONLY.
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame with SLEEP intervals, must have bedtime, waketime columns
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
    
    # ⭐ FILTER TO NIGHTTIME SLEEP ONLY (>180 minutes = 3 hours)
    nighttime_sleep = sleep_df[sleep_df['sleep_time'] >= 180].copy()
    
    if len(nighttime_sleep) == 0:
        print(f"Warning: No nighttime sleep found for {subject_id}")
        return None, None
    
    # Get unique dates
    dates = nighttime_sleep['start_date'].unique()
    dates = sorted(dates)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, len(dates) * 0.4)))
    
    # Color for nighttime sleep
    color = '#2E86AB'
    
    # Find min and max hours to set appropriate x-axis limits
    all_bed_hours = []
    all_wake_hours = []
    
    # Plot each sleep period
    for i, date in enumerate(dates):
        day_sleep = nighttime_sleep[nighttime_sleep['start_date'] == date]
        
        for _, row in day_sleep.iterrows():
            # Convert times to hours for plotting
            bed_hour = row['bedtime'].hour + row['bedtime'].minute / 60
            wake_hour = row['waketime'].hour + row['waketime'].minute / 60
            
            all_bed_hours.append(bed_hour)
            all_wake_hours.append(wake_hour)
            
            # Handle overnight sleep (crosses midnight)
            if wake_hour < bed_hour:
                # Split into two bars
                # Part 1: bedtime to midnight
                duration1 = 24 - bed_hour
                ax.barh(i, duration1, left=bed_hour, height=0.8,
                       color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Part 2: midnight to wake time
                ax.barh(i, wake_hour, left=0, height=0.8,
                       color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
            else:
                # Single bar (daytime sleep that doesn't cross midnight)
                duration = wake_hour - bed_hour
                ax.barh(i, duration, left=bed_hour, height=0.8,
                       color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Formatting
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates)
    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(f'Sleep Raster Plot - Nighttime Sleep Only - {subject_id}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # ⭐ DYNAMIC X-AXIS: Set based on actual sleep times
    # Find earliest bedtime and latest wake time
    min_bed = min(all_bed_hours)
    max_wake = max(all_wake_hours)
    
    # Adjust for overnight sleep
    # Typical: bedtime around 18-23 (6 PM - 11 PM), wake around 0-10 (12 AM - 10 AM)
    if max_wake < 12:  # Wake time is in morning (AM)
        # Extend to show full overnight period
        x_min = max(0, min_bed - 1)  # Start 1 hour before earliest bedtime
        x_max = min(36, max_wake + 25)  # Extend to show wake time (add 24 for next day + 1 padding)
        
        # Create tick labels
        hour_ticks = []
        hour_labels = []
        
        # Evening hours (6 PM onwards)
        for h in range(int(x_min), 24):
            if h % 3 == 0:  # Every 3 hours
                hour_ticks.append(h)
                pm_hour = h - 12 if h > 12 else h
                hour_labels.append(f'{pm_hour} PM' if h >= 12 else f'{h} AM')
        
        # Morning hours (12 AM onwards)
        for h in range(0, int(max_wake) + 2):
            adjusted_h = h + 24  # Add 24 for plotting
            if h % 3 == 0:
                hour_ticks.append(adjusted_h)
                hour_labels.append(f'{h if h != 0 else 12} AM')
        
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels, rotation=0)
    else:
        # Daytime sleep (unusual but handle it)
        x_min = max(0, min_bed - 1)
        x_max = min(24, max_wake + 1)
        ax.set_xlim(x_min, x_max)
        
        # Standard 24-hour ticks
        hour_ticks = list(range(int(x_min), int(x_max) + 1, 3))
        hour_labels = []
        for h in hour_ticks:
            if h == 0 or h == 24:
                hour_labels.append('12 AM')
            elif h < 12:
                hour_labels.append(f'{h} AM')
            elif h == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{h-12} PM')
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels)
    
    # Add vertical line for midnight
    ax.axvline(x=24, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='Midnight')
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


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
    Create scatter plot showing bedtime and wake time consistency.
    
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
    
    # Filter to nighttime sleep only (>180 min)
    nighttime = sleep_df[sleep_df['sleep_time'] >= 180].copy()
    
    if len(nighttime) == 0:
        print(f"No nighttime sleep found for {subject_id}")
        return None, None
    
    # Extract times as hours
    nighttime['bedtime_hour'] = nighttime['bedtime'].dt.hour + nighttime['bedtime'].dt.minute / 60
    nighttime['waketime_hour'] = nighttime['waketime'].dt.hour + nighttime['waketime'].dt.minute / 60
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dates = pd.to_datetime(nighttime['start_date'])
    
    # Plot bedtime and wake time
    ax.scatter(dates, nighttime['bedtime_hour'], s=100, alpha=0.7,
              color='#2E86AB', marker='o', label='Bedtime', edgecolors='white', linewidth=1.5)
    ax.scatter(dates, nighttime['waketime_hour'], s=100, alpha=0.7,
              color='#F18F01', marker='s', label='Wake Time', edgecolors='white', linewidth=1.5)
    
    # Add mean lines
    mean_bedtime = nighttime['bedtime_hour'].mean()
    mean_waketime = nighttime['waketime_hour'].mean()
    
    ax.axhline(y=mean_bedtime, color='#2E86AB', linestyle='--', alpha=0.5,
              linewidth=2, label=f'Mean Bedtime ({mean_bedtime:.1f}h)')
    ax.axhline(y=mean_waketime, color='#F18F01', linestyle='--', alpha=0.5,
              linewidth=2, label=f'Mean Wake Time ({mean_waketime:.1f}h)')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (24-hour)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sleep Timing Consistency - {subject_id}', fontsize=14, fontweight='bold', pad=20)
    
    # Y-axis: times
    y_ticks = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    y_labels = ['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(0, 24)
    
    # X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    ax.legend(loc='best', fontsize=10)
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