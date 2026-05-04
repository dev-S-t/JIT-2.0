"""
Micro-Expiry Visualization - Final Version

Extensions at specific days: 44, 51, 62
Extensions reduce shortage risk in following peaks.
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
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.linewidth'] = 0.3  # Very thin axis lines
plt.rcParams['axes.spines.top'] = False  # Remove top border
plt.rcParams['axes.spines.right'] = False  # Remove right border

# Heading color (blue-tinted)
HEADING_COLOR = '#1A5276'  # Dark blue


def create_comparison_figure():
    """Create JIT vs JIT+Micro comparison with manual extension placement."""
    
    # Load REAL data
    predictions_path = os.path.join(OUTPUT_PATH, '..', 'models', 'trained_models', 'predictions.csv')
    pred_df = pd.read_csv(predictions_path)
    
    # Select a 30-day window
    start, end = 40, 70
    days = np.arange(start, end)
    actual = pred_df['actual'].values[start:end]
    predicted = pred_df['pred_sarima'].values[start:end]
    
    # MANUAL extension placement as specified by user
    # Day 44 -> covers peaks 1 & 2
    # Day 51 -> covers peaks 3 & nearby bump
    # Day 62 -> covers peaks 6 & 7
    extension_days = [44, 51, 62]
    
    # Calculate extension heights (DOUBLED as per user request)
    max_gap = np.max(actual - predicted)
    extension_height = (max_gap + 5) * 2  # Double height
    
    # Create figure (not sharing x to show ticks on both)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('How Micro-Expiry Reduces Shortage Risk', fontsize=18, fontweight='bold', y=0.98, color=HEADING_COLOR)
    
    # Colors
    GREEN = '#81C784'
    RED = '#EF5350'
    BLUE_LINE = '#29B6F6'
    BLUE_DASH = '#1565C0'
    PURPLE = '#9C27B0'
    
    # ========== TOP: JIT ONLY ==========
    ax1.set_title('JIT Only — Prediction errors cause shortages', fontsize=14, fontweight='semibold', color=HEADING_COLOR)
    
    # Green fill under actual demand
    ax1.fill_between(days, 0, actual, color=GREEN, alpha=0.5, label='Demand')
    
    # Red fill - shortage gap
    ax1.fill_between(days, predicted, actual, 
                     where=(actual > predicted),
                     color=RED, alpha=0.7, label='Shortage Risk',
                     interpolate=True)
    
    # Lines
    ax1.plot(days, actual, color=BLUE_LINE, linewidth=2.5, label='Actual Demand')
    ax1.plot(days, predicted, color=BLUE_DASH, linewidth=2, linestyle=(0, (5, 3)), label='Prediction')  # Wider dash spacing
    
    ax1.set_ylabel('Units', fontweight='semibold', color=HEADING_COLOR)
    ax1.set_ylim(0, max(actual) * 1.2)
    ax1.legend(loc='upper right', frameon=False, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)  # Dotted grid both axes
    ax1.tick_params(colors='#555555')
    ax1.set_xlim(40, 70)
    
    # ========== BOTTOM: JIT + MICRO ==========
    # Title positioned inside the plot area (moved down)
    ax2.text(0.5, 0.92, 'JIT + Micro-Expiry — Extensions reduce shortage', 
             transform=ax2.transAxes, fontsize=14, fontweight='semibold', 
             color=HEADING_COLOR, ha='center', va='top')
    
    # Apply extension effect to prediction
    # Each extension boosts prediction for ~3-4 days after
    buffered_pred = predicted.copy().astype(float)
    
    for ext_day in extension_days:
        ext_idx = ext_day - start  # Convert to array index
        
        # Day 51 extension: reduced boost (prediction went too high)
        if ext_day == 51:
            boost_amount = max_gap * 0.4  # Much smaller boost for day 51
        else:
            boost_amount = max_gap * 0.8  # Normal boost for other days
        
        # Extension effect lasts 4-5 days
        for j in range(ext_idx, min(ext_idx + 5, len(buffered_pred))):
            # Gradual decay of effect
            decay = 1.0 - (j - ext_idx) * 0.15
            buffered_pred[j] += boost_amount * max(0.3, decay)
    
    # Green fill under actual demand
    ax2.fill_between(days, 0, actual, color=GREEN, alpha=0.5)
    
    # Red fill - REDUCED shortage gap (most peaks should be covered now)
    ax2.fill_between(days, buffered_pred, actual,
                     where=(actual > buffered_pred),
                     color=RED, alpha=0.7,
                     interpolate=True)
    
    # Lines
    ax2.plot(days, actual, color=BLUE_LINE, linewidth=2.5)
    ax2.plot(days, buffered_pred, color=BLUE_DASH, linewidth=2, linestyle=(0, (5, 3)))  # Wider dash spacing
    
    # Extension bars with labels - heights proportional to units
    # Bar positions shifted as per user request (visual only, doesn't affect prediction)
    bar_width = 0.8
    extension_units = {44: 8, 51: 5, 62: 7}  # Units extended
    bar_positions = {44: 44, 51: 50.5, 62: 62}  # Only 2nd bar shifted
    max_units = 8  # Reference for full height
    
    for ext_day in extension_days:
        units = extension_units[ext_day]
        bar_pos = bar_positions[ext_day]  # Use shifted position for bar
        bar_height = extension_height * (units / max_units)  # Proportional height
        
        ax2.bar(bar_pos, bar_height, width=bar_width, bottom=0,
                color=PURPLE, alpha=0.6, edgecolor=PURPLE, linewidth=2)
        # Add unit label on top
        ax2.text(bar_pos, bar_height + 1, f'{units}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color=PURPLE)
    
    ax2.set_xlabel('Simulation Day', fontweight='semibold', color=HEADING_COLOR)
    ax2.set_ylabel('Units', fontweight='semibold', color=HEADING_COLOR)
    ax2.set_ylim(0, max(actual) * 1.2)
    ax2.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)  # Dotted grid both axes
    ax2.tick_params(colors='#555555')
    
    # Legend
    extension_patch = mpatches.Patch(color=PURPLE, alpha=0.6, label='Extension')
    ax2.legend(handles=[extension_patch], loc='upper right', frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure9_jit_comparison.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure9_jit_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: figure9_jit_comparison.png/pdf")


if __name__ == "__main__":
    print("Generating Visualization...")
    create_comparison_figure()
    print("Done!")
