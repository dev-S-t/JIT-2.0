"""
Half-Year Model-Based Visualization
Real data from models - no manipulation
- Actual demand from data
- Predictions from SARIMA model
- Extensions from micro-expiry simulation
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_PATH = os.path.dirname(__file__)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

HEADING_COLOR = '#1A5276'


def generate_half_year_visualization():
    """Generate half-year visualization using real model data."""
    
    # Load data
    predictions_path = os.path.join(OUTPUT_PATH, '..', 'models', 'trained_models', 'predictions.csv')
    micro_path = os.path.join(OUTPUT_PATH, 'daily_jit_micro.csv')
    
    pred_df = pd.read_csv(predictions_path)
    micro_df = pd.read_csv(micro_path)
    
    # Use all data (full half year)
    days = np.arange(len(pred_df))
    actual = pred_df['actual'].values
    predicted = pred_df['pred_sarima'].values  # SARIMA model predictions
    extensions = micro_df['extensions_triggered'].values
    
    # Create protection effect (same logic as figure7)
    protection = np.zeros(len(days))
    for i, ext in enumerate(extensions):
        if ext > 0:
            if i+1 < len(protection):
                protection[i+1] += ext * 1.5
            if i+2 < len(protection):
                protection[i+2] += ext * 3.0
            if i+3 < len(protection):
                protection[i+3] += ext * 2.5
            if i+4 < len(protection):
                protection[i+4] += ext * 1.0
    
    # Calculate under-prediction area
    under_prediction = np.maximum(0, actual - predicted)
    protected_under = np.maximum(0, under_prediction - protection)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Colors - matching figure9 v2
    COLOR_DEMAND_FILL = '#90EE90'   # Light green
    COLOR_DEMAND_LINE = '#4FC3F7'   # Light blue
    COLOR_PREDICT_LINE = '#1565C0'  # Dark blue
    COLOR_DANGER = '#FF8A8A'        # Light red
    COLOR_INVESTMENT = '#BA68C8'    # Lighter purple
    
    # 1. Fill area under ACTUAL demand (green)
    ax.fill_between(days, 0, actual, alpha=0.55, color=COLOR_DEMAND_FILL, 
                    label='Demand', zorder=1)
    
    # 2. Fill under-prediction gap (RED - shortage risk)
    ax.fill_between(days, predicted, actual, 
                    where=(actual > predicted),
                    alpha=0.75, color=COLOR_DANGER, label='Shortage Risk',
                    interpolate=True, zorder=2)
    
    # 3. Fill protected area (same green with dots)
    protection_visible = under_prediction - protected_under
    protection_visible = np.maximum(0, protection_visible)
    protected_upper = predicted + protection_visible
    protected_upper = np.minimum(protected_upper, actual)
    
    ax.fill_between(days, predicted, protected_upper,
                    where=(protection_visible > 0),
                    alpha=1.0, color=COLOR_DEMAND_FILL, zorder=4,
                    interpolate=True, hatch='..', edgecolor=COLOR_DANGER, linewidth=0.5)
    
    # 4. Plot actual demand line
    ax.plot(days, actual, color=COLOR_DEMAND_LINE, linewidth=2, 
            label='Actual Demand', zorder=5)
    
    # 5. Plot prediction line
    ax.plot(days, predicted, color=COLOR_PREDICT_LINE, linewidth=1.5, 
            linestyle=(0, (5, 3)), label='Prediction', zorder=5)
    
    # 6. Extension bars - starting from 0
    extension_mask = extensions > 0
    if np.any(extension_mask):
        extension_heights = extensions[extension_mask] * 3 + 8
        extension_positions = days[extension_mask]
        
        ax.bar(extension_positions, extension_heights, 
               bottom=0,
               width=1.2, color=COLOR_INVESTMENT, alpha=0.85, 
               label='Extension', zorder=10, 
               edgecolor=COLOR_INVESTMENT, linewidth=1.5)
    
    # Labels and title - heading inside plot
    ax.set_xlabel('Simulation Day', fontweight='semibold', color=HEADING_COLOR)
    ax.set_ylabel('Platelet Units', fontweight='semibold', color=HEADING_COLOR)
    
    # Main heading inside the bounding box
    ax.text(0.5, 0.97, 'Half-Year Model Results', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            color=HEADING_COLOR, ha='center', va='top')
    ax.text(0.5, 0.91, 'Actual Demand vs SARIMA Predictions with Micro-Expiry Extensions', 
            transform=ax.transAxes, fontsize=12, fontweight='semibold', 
            color=HEADING_COLOR, ha='center', va='top')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_DEMAND_FILL, alpha=0.55, label='Demand'),
        mpatches.Patch(facecolor=COLOR_DANGER, alpha=0.75, label='Shortage Risk'),
        mpatches.Patch(facecolor=COLOR_DEMAND_FILL, alpha=1.0, hatch='..', edgecolor=COLOR_DANGER, label='Protected by Extension'),
        plt.Line2D([0], [0], color=COLOR_DEMAND_LINE, linewidth=2, label='Actual Demand'),
        plt.Line2D([0], [0], color=COLOR_PREDICT_LINE, linewidth=1.5, linestyle='--', label='Prediction'),
        mpatches.Patch(facecolor=COLOR_INVESTMENT, alpha=0.85, label='Extension'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)
    
    ax.set_xlim(0, len(days))
    ax.set_ylim(0, max(actual.max(), predicted.max()) * 1.25)
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.tick_params(colors='#555555')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure_half_year_model.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure_half_year_model.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: figure_half_year_model.png/pdf")
    print(f"  Data range: {pred_df['date'].iloc[0]} to {pred_df['date'].iloc[-1]}")
    print(f"  Total days: {len(days)}")
    print(f"  Extensions triggered: {np.sum(extensions > 0)} days")


if __name__ == "__main__":
    print("Generating Half-Year Model Visualization...")
    generate_half_year_visualization()
    print("Done!")
