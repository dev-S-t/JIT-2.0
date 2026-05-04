"""
Platelet Demand Data Visualization
Creates line graphs showing temporal patterns in the synthetic data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load the generated datasets."""
    hamilton = pd.read_csv('platelet_demand_hamilton_medium_hospital.csv', parse_dates=['date'])
    stanford = pd.read_csv('platelet_demand_stanford_large_hospital.csv', parse_dates=['date'])
    return hamilton, stanford

def plot_daily_demand(hamilton, stanford):
    """Plot daily demand over time for both hospitals."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Hamilton
    axes[0].plot(hamilton['date'], hamilton['units_demanded'], 
                 color='#2E86AB', alpha=0.7, linewidth=0.8, label='Daily Demand')
    axes[0].axhline(y=hamilton['units_demanded'].mean(), color='#E94F37', 
                    linestyle='--', linewidth=2, label=f'Mean: {hamilton["units_demanded"].mean():.1f}')
    axes[0].set_ylabel('Units Demanded')
    axes[0].set_title('Hamilton Medium Hospital - Daily Platelet Demand (2024-2025)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, hamilton['units_demanded'].max() * 1.1)
    
    # Stanford
    axes[1].plot(stanford['date'], stanford['units_demanded'], 
                 color='#7B2D8E', alpha=0.7, linewidth=0.8, label='Daily Demand')
    axes[1].axhline(y=stanford['units_demanded'].mean(), color='#E94F37', 
                    linestyle='--', linewidth=2, label=f'Mean: {stanford["units_demanded"].mean():.1f}')
    axes[1].set_ylabel('Units Demanded')
    axes[1].set_title('Stanford Large Hospital - Daily Platelet Demand (2024-2025)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, stanford['units_demanded'].max() * 1.1)
    
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plot_daily_demand.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_daily_demand.png")

def plot_weekly_pattern(hamilton, stanford):
    """Plot average demand by day of week."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#E94F37', '#7B2D8E', '#7B2D8E']
    
    # Hamilton
    hamilton_by_day = hamilton.groupby('day_name')['units_demanded'].mean().reindex(days)
    bars1 = axes[0].bar(days, hamilton_by_day, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=hamilton['units_demanded'].mean(), color='gray', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Average Units Demanded')
    axes[0].set_title('Hamilton - Weekly Pattern', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, hamilton_by_day):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Stanford
    stanford_by_day = stanford.groupby('day_name')['units_demanded'].mean().reindex(days)
    bars2 = axes[1].bar(days, stanford_by_day, color=colors, edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=stanford['units_demanded'].mean(), color='gray', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Average Units Demanded')
    axes[1].set_title('Stanford - Weekly Pattern', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, stanford_by_day):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot_weekly_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_weekly_pattern.png")

def plot_monthly_pattern(hamilton, stanford):
    """Plot average demand by month (seasonality)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    hamilton_by_month = hamilton.groupby('month_name')['units_demanded'].mean().reindex(months)
    stanford_by_month = stanford.groupby('month_name')['units_demanded'].mean().reindex(months)
    
    x = range(len(months))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], hamilton_by_month, width, 
                   label='Hamilton', color='#2E86AB', edgecolor='white')
    bars2 = ax.bar([i + width/2 for i in x], stanford_by_month, width, 
                   label='Stanford', color='#7B2D8E', edgecolor='white')
    
    ax.set_ylabel('Average Units Demanded')
    ax.set_title('Monthly Seasonality - Platelet Demand', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3] for m in months])
    ax.legend()
    
    # Highlight Dec/Jan slump
    ax.axvspan(-0.5, 1.5, alpha=0.15, color='red', label='Winter Slump')
    ax.text(0.5, ax.get_ylim()[1]*0.95, 'Winter\nSlump', ha='center', fontsize=9, color='#E94F37')
    
    plt.tight_layout()
    plt.savefig('plot_monthly_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_monthly_seasonality.png")

def plot_rolling_average(hamilton, stanford):
    """Plot 7-day rolling average to smooth out daily noise."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    hamilton_roll = hamilton.set_index('date')['units_demanded'].rolling(window=7).mean()
    stanford_roll = stanford.set_index('date')['units_demanded'].rolling(window=7).mean()
    
    ax.plot(hamilton_roll.index, hamilton_roll, color='#2E86AB', linewidth=2, 
            label='Hamilton (7-day avg)', alpha=0.9)
    ax.plot(stanford_roll.index, stanford_roll, color='#7B2D8E', linewidth=2, 
            label='Stanford (7-day avg)', alpha=0.9)
    
    ax.set_ylabel('Units Demanded (7-day Rolling Average)')
    ax.set_title('Platelet Demand Trends - 7-Day Rolling Average', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plot_rolling_average.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_rolling_average.png")

def main():
    print("="*60)
    print("PLATELET DEMAND DATA VISUALIZATION")
    print("="*60)
    
    hamilton, stanford = load_data()
    print(f"\n📊 Loaded {len(hamilton)} records for Hamilton")
    print(f"📊 Loaded {len(stanford)} records for Stanford\n")
    
    print("Generating plots...")
    plot_daily_demand(hamilton, stanford)
    plot_weekly_pattern(hamilton, stanford)
    plot_monthly_pattern(hamilton, stanford)
    plot_rolling_average(hamilton, stanford)
    
    print("\n" + "="*60)
    print("✅ ALL PLOTS GENERATED!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. plot_daily_demand.png")
    print("  2. plot_weekly_pattern.png")
    print("  3. plot_monthly_seasonality.png")
    print("  4. plot_rolling_average.png")

if __name__ == "__main__":
    main()
