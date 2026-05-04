"""
Micro-Expiry Support Visualization

Shows how the extension (investment) absorbs prediction errors:
- Solid line: Actual demand (light blue fill)
- Dotted line: JIT prediction (light red fill where under-predicted)
- Vertical bars: Extension investments
- Effect: Red area reduces after extension takes effect
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

# Heading color (blue-tinted) - same as figure9
HEADING_COLOR = '#1A5276'


def generate_micro_expiry_visualization():
    """Generate the Micro-Expiry Support visualization."""
    
    # Load data
    predictions_path = os.path.join(OUTPUT_PATH, '..', 'models', 'trained_models', 'predictions.csv')
    micro_path = os.path.join(OUTPUT_PATH, 'daily_jit_micro.csv')
    
    pred_df = pd.read_csv(predictions_path)
    micro_df = pd.read_csv(micro_path)
    
    # Get a 30-day window with interesting extension activity
    extension_days = micro_df[micro_df['extensions_triggered'] > 0]['day'].values
    if len(extension_days) > 0:
        center_day = extension_days[len(extension_days)//2]
        start_day = max(0, center_day - 15)
        end_day = min(len(micro_df), center_day + 20)
    else:
        start_day = 50
        end_day = 85
    
    # Slice data
    days = np.arange(start_day, end_day)
    actual = pred_df['actual'].values[start_day:end_day]
    predicted = pred_df['pred_sarima'].values[start_day:end_day]
    extensions = micro_df['extensions_triggered'].values[start_day:end_day]
    
    # Create protection effect
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
    
    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color scheme (user requested)
    COLOR_DEMAND_FILL = '#90EE90'      # Light green - demand met
    COLOR_DEMAND_LINE = '#4FC3F7'      # Light blue - actual demand line
    COLOR_PREDICT_LINE = '#1565C0'     # Dark blue - prediction dashes
    COLOR_DANGER = '#E53935'           # Danger red - under-prediction gap
    COLOR_PROTECTED = '#90EE90'        # Same green as demand - more opaque
    COLOR_INVESTMENT = '#9C27B0'       # Purple - extension investment (see-through)
    
    # 1. Fill area under ACTUAL demand (green - full demand area)
    ax.fill_between(days, 0, actual, alpha=0.4, color=COLOR_DEMAND_FILL, 
                    label='Demand', zorder=1)
    
    # 2. Fill under-prediction gap (RED - only where actual > predicted, layered on top)
    ax.fill_between(days, predicted, actual, 
                    where=(actual > predicted),
                    alpha=0.6, color=COLOR_DANGER, label='Shortage Risk',
                    interpolate=True, zorder=2)
    
    # 3. Fill protected area (same green but more opaque - bounded by actual demand)
    protection_visible = under_prediction - protected_under
    protection_visible = np.maximum(0, protection_visible)
    
    # Calculate upper bound - protected area should not exceed actual demand
    protected_upper = predicted + protection_visible
    protected_upper = np.minimum(protected_upper, actual)  # Bound by actual demand
    
    ax.fill_between(days, predicted, protected_upper,
                    where=(protection_visible > 0),
                    alpha=1.0, color=COLOR_DEMAND_FILL, zorder=4,
                    interpolate=True, hatch='..', edgecolor=COLOR_DANGER, linewidth=0.5)
    
    # 4. Plot actual demand line (light blue, thicker)
    ax.plot(days, actual, color=COLOR_DEMAND_LINE, linewidth=3, 
            label='Actual Demand', zorder=5, solid_capstyle='round')
    
    # 5. Plot JIT prediction line (dark blue dashes, thicker)
    ax.plot(days, predicted, color=COLOR_PREDICT_LINE, linewidth=2.5, 
            linestyle='--', dashes=(5, 3), label='JIT Prediction', zorder=5)
    
    # 6. Vertical bars for extensions - starting from 0
    bar_width = 0.8
    extension_mask = extensions > 0
    if np.any(extension_mask):
        extension_heights = extensions[extension_mask] * 4 + 5  # Add base height
        extension_positions = days[extension_mask]
        
        ax.bar(extension_positions, extension_heights, 
               bottom=0,  # Start from 0
               width=bar_width, color=COLOR_INVESTMENT, alpha=0.7, 
               label='Extension', zorder=10, 
               edgecolor=COLOR_INVESTMENT, linewidth=2)
        
        # Add extension count labels - manually specified as 9, 9, 18, 18
        manual_labels = [9, 9, 18, 18]  # User-specified labels
        for idx, (pos, height) in enumerate(zip(extension_positions, extension_heights)):
            label = manual_labels[idx] if idx < len(manual_labels) else ''
            ax.text(pos, height + 1, f'{label}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color=COLOR_INVESTMENT, zorder=11)
    
    # Labels and title - heading inside plot
    ax.set_xlabel('Simulation Day', fontweight='semibold', color=HEADING_COLOR)
    ax.set_ylabel('Platelet Units', fontweight='semibold', color=HEADING_COLOR)
    
    # Main heading inside the bounding box
    ax.text(0.5, 0.97, 'Micro-Expiry Support in Action', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            color=HEADING_COLOR, ha='center', va='top')
    ax.text(0.5, 0.91, 'How Extensions Absorb Prediction Errors', 
            transform=ax.transAxes, fontsize=13, fontweight='semibold', 
            color=HEADING_COLOR, ha='center', va='top')
    
    # Updated legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_DEMAND_FILL, alpha=0.4, label='Demand'),
        mpatches.Patch(facecolor=COLOR_DANGER, alpha=0.6, label='Shortage Risk'),
        mpatches.Patch(facecolor=COLOR_PROTECTED, alpha=0.5, hatch='..', edgecolor=COLOR_DANGER, label='Protected by Extension'),
        plt.Line2D([0], [0], color=COLOR_DEMAND_LINE, linewidth=3, label='Actual Demand'),
        plt.Line2D([0], [0], color=COLOR_PREDICT_LINE, linewidth=2.5, linestyle='--', label='Prediction'),
        mpatches.Patch(facecolor=COLOR_INVESTMENT, alpha=0.7, label='Extension'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)
    
    # Annotation - moved slightly
    if len(extension_days) > 0 and np.any(extension_mask):
        first_ext = days[extension_mask][0]
        ax.annotate('Extension triggered\n→ Protection in 1-2 days', 
                    xy=(first_ext, 12),
                    xytext=(first_ext + 8, 35),
                    fontsize=10, fontweight='bold', color=HEADING_COLOR,
                    arrowprops=dict(arrowstyle='->', color=HEADING_COLOR, lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=1.0, edgecolor=HEADING_COLOR),
                    zorder=15)
    
    ax.set_xlim(days[0]-0.5, days[-1]+0.5)
    ax.set_ylim(0, max(actual.max(), predicted.max()) * 1.3)
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.tick_params(colors='#555555')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure7_micro_expiry_action.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure7_micro_expiry_action.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: figure7_micro_expiry_action.png/pdf")


def generate_full_timeline_view():
    """Generate a full timeline view with improved colors."""
    
    # Load data
    predictions_path = os.path.join(OUTPUT_PATH, '..', 'models', 'trained_models', 'predictions.csv')
    micro_path = os.path.join(OUTPUT_PATH, 'daily_jit_micro.csv')
    
    pred_df = pd.read_csv(predictions_path)
    micro_df = pd.read_csv(micro_path)
    
    # Use full dataset
    days = np.arange(len(pred_df))
    actual = pred_df['actual'].values
    predicted = pred_df['pred_sarima'].values
    extensions = micro_df['extensions_triggered'].values
    
    # Color scheme (same as figure 7)
    COLOR_DEMAND_FILL = '#90EE90'      # Light green - demand met
    COLOR_DEMAND_LINE = '#4FC3F7'      # Light blue - actual demand line
    COLOR_PREDICT_LINE = '#1565C0'     # Dark blue - prediction dashes
    COLOR_DANGER = '#E53935'           # Danger red - under-prediction gap
    COLOR_INVESTMENT = '#9C27B0'       # Purple - extension investment
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 1. Fill demand met area (light green)
    demand_met = np.minimum(actual, predicted)
    ax.fill_between(days, 0, demand_met, alpha=0.4, color=COLOR_DEMAND_FILL)
    
    # 2. Fill under-prediction gap (danger red)
    ax.fill_between(days, predicted, actual, 
                    where=(actual > predicted),
                    alpha=0.5, color=COLOR_DANGER, label='Under-prediction Gap')
    
    # 3. Plot actual demand line (light blue)
    ax.plot(days, actual, color=COLOR_DEMAND_LINE, linewidth=2, 
            label='Actual Demand', zorder=3)
    
    # 4. Plot JIT prediction line (dark blue dashes)
    ax.plot(days, predicted, color=COLOR_PREDICT_LINE, linewidth=1.5, linestyle='--', 
            alpha=0.9, label='JIT Prediction', zorder=3)
    
    # 5. Vertical bars for extensions (see-through purple)
    extension_mask = extensions > 0
    if np.any(extension_mask):
        extension_heights = extensions[extension_mask] * 3 + max(actual) * 0.5
        extension_positions = days[extension_mask]
        
        ax.bar(extension_positions, extension_heights,
               bottom=0,
               width=1.5, color=COLOR_INVESTMENT, alpha=0.4, 
               label='Extension Investment', zorder=5, edgecolor=COLOR_INVESTMENT, linewidth=1.5)
    
    ax.set_xlabel('Simulation Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Platelet Units', fontsize=13, fontweight='bold')
    ax.set_title('Full Timeline: Demand, Prediction, and Micro-Expiry Investments', 
                 fontsize=15, fontweight='bold', pad=10)
    
    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_DEMAND_FILL, alpha=0.5, label='Demand Met'),
        plt.Line2D([0], [0], color=COLOR_DEMAND_LINE, linewidth=2, label='Actual Demand'),
        plt.Line2D([0], [0], color=COLOR_PREDICT_LINE, linewidth=1.5, linestyle='--', label='JIT Prediction'),
        mpatches.Patch(facecolor=COLOR_DANGER, alpha=0.5, label='Under-prediction Gap'),
        mpatches.Patch(facecolor=COLOR_INVESTMENT, alpha=0.4, label='Extension Investment'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlim(0, len(days))
    ax.set_ylim(0, max(actual.max(), predicted.max()) * 1.4)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure8_full_timeline.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure8_full_timeline.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: figure8_full_timeline.png/pdf")


if __name__ == "__main__":
    print("Generating Micro-Expiry Support Visualizations...")
    generate_micro_expiry_visualization()
    generate_full_timeline_view()
    print("Done!")
