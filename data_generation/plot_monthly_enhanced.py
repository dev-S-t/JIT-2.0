"""
Enhanced Monthly Seasonality Visualization
Shows monthly patterns with better Y-axis scaling to make differences visible
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

def plot_monthly_enhanced():
    """Create enhanced monthly seasonality plots with better visibility."""
    
    hamilton = pd.read_csv('platelet_demand_hamilton_medium_hospital.csv', parse_dates=['date'])
    stanford = pd.read_csv('platelet_demand_stanford_large_hospital.csv', parse_dates=['date'])
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    hamilton_by_month = hamilton.groupby('month')['units_demanded'].mean()
    stanford_by_month = stanford.groupby('month')['units_demanded'].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========== TOP LEFT: Hamilton Line Plot with Zoomed Y-axis ==========
    ax1 = axes[0, 0]
    ax1.plot(months, hamilton_by_month.values, 'o-', color='#2E86AB', linewidth=2.5, markersize=10)
    ax1.axhline(y=hamilton['units_demanded'].mean(), color='#E94F37', linestyle='--', 
                linewidth=2, label=f'Overall Mean: {hamilton["units_demanded"].mean():.1f}')
    ax1.fill_between(months, hamilton_by_month.values, hamilton['units_demanded'].mean(), 
                     alpha=0.3, color='#2E86AB')
    
    # Key: Y-axis starts near the minimum, not at 0
    y_min = hamilton_by_month.min() - 1
    y_max = hamilton_by_month.max() + 1
    ax1.set_ylim(y_min, y_max)
    
    ax1.set_ylabel('Avg Units Demanded', fontsize=11)
    ax1.set_title('Hamilton - Monthly Seasonality (Zoomed)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Add value labels
    for i, (m, v) in enumerate(zip(months, hamilton_by_month.values)):
        ax1.annotate(f'{v:.1f}', (i, v), textcoords="offset points", 
                     xytext=(0, 8), ha='center', fontsize=9)
    
    # ========== TOP RIGHT: Stanford Line Plot with Zoomed Y-axis ==========
    ax2 = axes[0, 1]
    ax2.plot(months, stanford_by_month.values, 'o-', color='#7B2D8E', linewidth=2.5, markersize=10)
    ax2.axhline(y=stanford['units_demanded'].mean(), color='#E94F37', linestyle='--', 
                linewidth=2, label=f'Overall Mean: {stanford["units_demanded"].mean():.1f}')
    ax2.fill_between(months, stanford_by_month.values, stanford['units_demanded'].mean(),
                     alpha=0.3, color='#7B2D8E')
    
    y_min = stanford_by_month.min() - 1
    y_max = stanford_by_month.max() + 1
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_ylabel('Avg Units Demanded', fontsize=11)
    ax2.set_title('Stanford - Monthly Seasonality (Zoomed)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    for i, (m, v) in enumerate(zip(months, stanford_by_month.values)):
        ax2.annotate(f'{v:.1f}', (i, v), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9)
    
    # ========== BOTTOM LEFT: % Deviation from Mean (Hamilton) ==========
    ax3 = axes[1, 0]
    hamilton_pct = ((hamilton_by_month - hamilton['units_demanded'].mean()) / 
                    hamilton['units_demanded'].mean() * 100)
    colors = ['#E94F37' if x < 0 else '#2E86AB' for x in hamilton_pct.values]
    bars = ax3.bar(months, hamilton_pct.values, color=colors, edgecolor='white', linewidth=1.5)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylabel('% Deviation from Mean', fontsize=11)
    ax3.set_title('Hamilton - Monthly Deviation (%)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, hamilton_pct.values):
        y_offset = 0.5 if val >= 0 else -1.5
        ax3.text(bar.get_x() + bar.get_width()/2, val + y_offset, 
                 f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # ========== BOTTOM RIGHT: % Deviation from Mean (Stanford) ==========
    ax4 = axes[1, 1]
    stanford_pct = ((stanford_by_month - stanford['units_demanded'].mean()) / 
                    stanford['units_demanded'].mean() * 100)
    colors = ['#E94F37' if x < 0 else '#7B2D8E' for x in stanford_pct.values]
    bars = ax4.bar(months, stanford_pct.values, color=colors, edgecolor='white', linewidth=1.5)
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.set_ylabel('% Deviation from Mean', fontsize=11)
    ax4.set_title('Stanford - Monthly Deviation (%)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, stanford_pct.values):
        y_offset = 0.5 if val >= 0 else -1.5
        ax4.text(bar.get_x() + bar.get_width()/2, val + y_offset,
                 f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    plt.suptitle('Monthly Seasonality Analysis - Platelet Demand', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plot_monthly_enhanced.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plot_monthly_enhanced.png")


if __name__ == "__main__":
    print("Generating enhanced monthly seasonality plot...")
    plot_monthly_enhanced()
    print("Done!")
