"""
Multi-Area Multi-Year Visualization
4 graphs: 2 areas (Hamilton, Stanford) x 2 years (2024, 2025)
Each graph for full year of data with predictions and extensions
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from scipy.ndimage import uniform_filter1d

OUTPUT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(OUTPUT_PATH, '..', 'data')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

HEADING_COLOR = '#1A5276'


def generate_predictions(actual, window=7):
    """Generate simple moving average prediction (simulates SARIMA)."""
    # Shift by 1 to simulate prediction from yesterday
    predictions = uniform_filter1d(actual.astype(float), size=window, mode='nearest')
    predictions = np.roll(predictions, 1)
    predictions[0] = actual[0]  # First day uses actual
    return predictions


def simulate_extensions(actual, predicted):
    """Simulate extension decisions based on prediction under-performance."""
    extensions = np.zeros(len(actual))
    cumulative_under = 0
    
    for i in range(len(actual)):
        under = max(0, actual[i] - predicted[i])
        cumulative_under += under * 0.3
        
        # Trigger extension when cumulative under-prediction exceeds threshold
        if cumulative_under > 15 and i > 7:
            extensions[i] = int(cumulative_under / 8)
            cumulative_under = cumulative_under * 0.2  # Reset partially
    
    return extensions


def create_year_visualization(df, area_name, year, ax):
    """Create visualization for one area and one year."""
    
    # Filter to specific year
    df['date'] = pd.to_datetime(df['date'])
    year_df = df[df['date'].dt.year == year].copy()
    
    if len(year_df) == 0:
        ax.text(0.5, 0.5, f'No data for {year}', transform=ax.transAxes, ha='center')
        return
    
    days = np.arange(len(year_df))
    actual = year_df['units_demanded'].values
    predicted = generate_predictions(actual, window=7)
    extensions = simulate_extensions(actual, predicted)
    
    # Protection effect
    protection = np.zeros(len(days))
    for i, ext in enumerate(extensions):
        if ext > 0:
            for offset, weight in [(1, 1.5), (2, 3.0), (3, 2.5), (4, 1.0)]:
                if i + offset < len(protection):
                    protection[i + offset] += ext * weight
    
    under_prediction = np.maximum(0, actual - predicted)
    protected_under = np.maximum(0, under_prediction - protection)
    
    # Colors
    COLOR_DEMAND_FILL = '#90EE90'
    COLOR_DEMAND_LINE = '#4FC3F7'
    COLOR_PREDICT_LINE = '#1565C0'
    COLOR_DANGER = '#FF8A8A'
    COLOR_INVESTMENT = '#BA68C8'
    
    # Plot fills
    ax.fill_between(days, 0, actual, alpha=0.55, color=COLOR_DEMAND_FILL, zorder=1)
    
    ax.fill_between(days, predicted, actual, 
                    where=(actual > predicted),
                    alpha=0.75, color=COLOR_DANGER,
                    interpolate=True, zorder=2)
    
    # Protected area
    protection_visible = under_prediction - protected_under
    protection_visible = np.maximum(0, protection_visible)
    protected_upper = predicted + protection_visible
    protected_upper = np.minimum(protected_upper, actual)
    
    ax.fill_between(days, predicted, protected_upper,
                    where=(protection_visible > 0),
                    alpha=1.0, color=COLOR_DEMAND_FILL, zorder=4,
                    interpolate=True, hatch='..', edgecolor=COLOR_DANGER, linewidth=0.5)
    
    # Lines
    ax.plot(days, actual, color=COLOR_DEMAND_LINE, linewidth=1.5, zorder=5)
    ax.plot(days, predicted, color=COLOR_PREDICT_LINE, linewidth=1.2, 
            linestyle=(0, (5, 3)), zorder=5)
    
    # Extension bars
    extension_mask = extensions > 0
    if np.any(extension_mask):
        extension_heights = extensions[extension_mask] * 2 + 5
        extension_positions = days[extension_mask]
        
        ax.bar(extension_positions, extension_heights, 
               bottom=0, width=2, color=COLOR_INVESTMENT, alpha=0.85, 
               edgecolor=COLOR_INVESTMENT, linewidth=1.2, zorder=10)
    
    # Title and labels
    ax.text(0.5, 0.97, f'{area_name.replace("_", " ")} - {year}', 
            transform=ax.transAxes, fontsize=14, fontweight='bold', 
            color=HEADING_COLOR, ha='center', va='top')
    
    ax.set_xlabel('Day of Year', fontweight='semibold', color=HEADING_COLOR, fontsize=10)
    ax.set_ylabel('Units', fontweight='semibold', color=HEADING_COLOR, fontsize=10)
    
    ax.set_xlim(0, len(days))
    ax.set_ylim(0, max(actual.max(), predicted.max()) * 1.2)
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.tick_params(colors='#555555', labelsize=9)
    
    # Stats text
    ext_count = np.sum(extensions > 0)
    shortage_days = np.sum(protected_under > 0)
    ax.text(0.98, 0.03, f'Extensions: {ext_count} | Shortage risk days: {shortage_days}', 
            transform=ax.transAxes, fontsize=9, color='#666666', 
            ha='right', va='bottom')


def generate_all_visualizations():
    """Generate all 4 individual visualizations with JIT comparison."""
    
    # Load data for both areas
    hamilton_df = pd.read_csv(os.path.join(DATA_PATH, 'platelet_demand_hamilton_medium_hospital.csv'))
    stanford_df = pd.read_csv(os.path.join(DATA_PATH, 'platelet_demand_stanford_large_hospital.csv'))
    
    areas = [
        ('Hamilton_Medium_Hospital', hamilton_df),
        ('Stanford_Large_Hospital', stanford_df)
    ]
    years = [2024, 2025]
    
    for area_name, df in areas:
        for year in years:
            # Create 2-panel figure: JIT+Micro vs JIT Only
            fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
            
            # Filter to specific year
            df['date'] = pd.to_datetime(df['date'])
            year_df = df[df['date'].dt.year == year].copy()
            
            if len(year_df) == 0:
                continue
            
            days = np.arange(len(year_df))
            actual = year_df['units_demanded'].values
            predicted = generate_predictions(actual, window=7)
            extensions = simulate_extensions(actual, predicted)
            
            # Calculate protection from extensions
            protection = np.zeros(len(days))
            for i, ext in enumerate(extensions):
                if ext > 0:
                    for offset, weight in [(1, 1.5), (2, 3.0), (3, 2.5), (4, 1.0)]:
                        if i + offset < len(protection):
                            protection[i + offset] += ext * weight
            
            under_prediction = np.maximum(0, actual - predicted)
            protected_under = np.maximum(0, under_prediction - protection)
            
            # Colors
            COLOR_DEMAND_FILL = '#90EE90'
            COLOR_DEMAND_LINE = '#4FC3F7'
            COLOR_PREDICT_LINE = '#1565C0'
            COLOR_DANGER = '#FF8A8A'
            COLOR_INVESTMENT = '#BA68C8'
            
            # ======= Panel 1: JIT + Micro-Expiry (with extensions) =======
            ax1 = axes[0]
            
            ax1.fill_between(days, 0, actual, alpha=0.55, color=COLOR_DEMAND_FILL, zorder=1)
            ax1.fill_between(days, predicted, actual, 
                            where=(actual > predicted),
                            alpha=0.75, color=COLOR_DANGER, interpolate=True, zorder=2)
            
            # Protected area
            protection_visible = under_prediction - protected_under
            protection_visible = np.maximum(0, protection_visible)
            protected_upper = predicted + protection_visible
            protected_upper = np.minimum(protected_upper, actual)
            
            ax1.fill_between(days, predicted, protected_upper,
                            where=(protection_visible > 0),
                            alpha=1.0, color=COLOR_DEMAND_FILL, zorder=4,
                            interpolate=True, hatch='..', edgecolor=COLOR_DANGER, linewidth=0.5)
            
            ax1.plot(days, actual, color=COLOR_DEMAND_LINE, linewidth=1.5, zorder=5)
            ax1.plot(days, predicted, color=COLOR_PREDICT_LINE, linewidth=1.2, 
                    linestyle=(0, (5, 3)), zorder=5)
            
            # Extension bars
            extension_mask = extensions > 0
            if np.any(extension_mask):
                extension_heights = extensions[extension_mask] * 2 + 5
                extension_positions = days[extension_mask]
                ax1.bar(extension_positions, extension_heights, 
                       bottom=0, width=2, color=COLOR_INVESTMENT, alpha=0.85, 
                       edgecolor=COLOR_INVESTMENT, linewidth=1.2, zorder=10)
            
            ax1.text(0.5, 0.97, 'JIT + Micro-Expiry System', 
                    transform=ax1.transAxes, fontsize=13, fontweight='bold', 
                    color=HEADING_COLOR, ha='center', va='top')
            
            ext_count = np.sum(extensions > 0)
            shortage_with_ext = np.sum(protected_under > 0)
            ax1.text(0.98, 0.03, f'Extensions: {ext_count} | Shortage risk: {shortage_with_ext} days', 
                    transform=ax1.transAxes, fontsize=10, color='#666666', ha='right', va='bottom')
            
            ax1.set_ylabel('Units', fontweight='semibold', color=HEADING_COLOR)
            ax1.set_ylim(0, max(actual.max(), predicted.max()) * 1.2)
            ax1.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
            ax1.tick_params(colors='#555555')
            
            # ======= Panel 2: JIT Only (no extensions) =======
            ax2 = axes[1]
            
            ax2.fill_between(days, 0, actual, alpha=0.55, color=COLOR_DEMAND_FILL, zorder=1)
            ax2.fill_between(days, predicted, actual, 
                            where=(actual > predicted),
                            alpha=0.75, color=COLOR_DANGER, interpolate=True, zorder=2)
            
            ax2.plot(days, actual, color=COLOR_DEMAND_LINE, linewidth=1.5, zorder=5)
            ax2.plot(days, predicted, color=COLOR_PREDICT_LINE, linewidth=1.2, 
                    linestyle=(0, (5, 3)), zorder=5)
            
            ax2.text(0.5, 0.97, 'JIT Only (No Extension)', 
                    transform=ax2.transAxes, fontsize=13, fontweight='bold', 
                    color=HEADING_COLOR, ha='center', va='top')
            
            shortage_jit_only = np.sum(under_prediction > 0)
            ax2.text(0.98, 0.03, f'Shortage risk: {shortage_jit_only} days (all unprotected)', 
                    transform=ax2.transAxes, fontsize=10, color='#666666', ha='right', va='bottom')
            
            ax2.set_xlabel('Day of Year', fontweight='semibold', color=HEADING_COLOR)
            ax2.set_ylabel('Units', fontweight='semibold', color=HEADING_COLOR)
            ax2.set_xlim(0, len(days))
            ax2.set_ylim(0, max(actual.max(), predicted.max()) * 1.2)
            ax2.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
            ax2.tick_params(colors='#555555')
            
            # Common legend
            legend_elements = [
                mpatches.Patch(facecolor=COLOR_DEMAND_FILL, alpha=0.55, label='Demand'),
                mpatches.Patch(facecolor=COLOR_DANGER, alpha=0.75, label='Shortage Risk'),
                mpatches.Patch(facecolor=COLOR_DEMAND_FILL, alpha=1.0, hatch='..', edgecolor=COLOR_DANGER, label='Protected'),
                plt.Line2D([0], [0], color=COLOR_DEMAND_LINE, linewidth=2, label='Actual'),
                plt.Line2D([0], [0], color=COLOR_PREDICT_LINE, linewidth=1.5, linestyle='--', label='Predicted'),
                mpatches.Patch(facecolor=COLOR_INVESTMENT, alpha=0.85, label='Extension'),
            ]
            fig.legend(handles=legend_elements, loc='upper center', ncol=6, 
                      fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.02))
            
            # Main title
            fig.suptitle(f'{area_name.replace("_", " ")} - {year}\nJIT + Micro-Expiry vs JIT Only Comparison', 
                        fontsize=16, fontweight='bold', color=HEADING_COLOR, y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.94])
            
            filename = f'figure_{area_name.lower()}_{year}_comparison'
            plt.savefig(os.path.join(OUTPUT_PATH, f'{filename}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_PATH, f'{filename}.pdf'), bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {filename}.png/pdf")


if __name__ == "__main__":
    print("Generating Multi-Area Multi-Year Comparison Visualizations...")
    generate_all_visualizations()
    print("\nDone! Generated 4 individual comparison figures.")
