"""
Estimated Savings Visualization - Styled Version
Matches the styling of figure7 and figure9
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_PATH = os.path.dirname(__file__)

# Style settings (matching figure7/figure9)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors matching other figures
HEADING_COLOR = '#1A5276'
COLOR_JIT_ONLY = '#4FC3F7'  # Light blue (demand line color)
COLOR_JIT_MICRO = '#90EE90'  # Green (demand fill color)
COLOR_DANGER = '#FF8A8A'  # Red for negative
COLOR_INVESTMENT = '#BA68C8'  # Purple (extension color)


def generate_savings_visualization():
    """Generate styled estimated savings bar chart."""
    
    # Data from simulation
    categories = ['Wastage\nReduction', 'Shortage\nReduction', 'Overall\nEfficiency']
    
    jit_only = [100, -100, 45]  # JIT-Only vs Traditional (% improvement)
    jit_micro = [100, 84, 95]   # JIT+Micro vs Traditional (% improvement)
    
    # Error bars (uncertainty range)
    jit_only_err = [5, 15, 8]
    jit_micro_err = [5, 10, 5]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bars with proper colors and error bars
    bars1 = ax.bar(x - width/2, jit_only, width, label='JIT-Only vs Traditional',
                   color=COLOR_JIT_ONLY, edgecolor=COLOR_JIT_ONLY, linewidth=1.5, alpha=0.85,
                   yerr=jit_only_err, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    bars2 = ax.bar(x + width/2, jit_micro, width, label='JIT+Micro vs Traditional',
                   color=COLOR_JIT_MICRO, edgecolor='#2ECC71', linewidth=1.5, alpha=0.85,
                   yerr=jit_micro_err, capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    
    # Keep consistent colors (don't change negative bars to red)
    # Both bar sets maintain their legend colors
    
    # Add value labels on bars (offset to clear error bars)
    def add_labels(bars, err_values):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            err = err_values[i]
            if height >= 0:
                label_y = height + err + 5  # Above the error bar
                va = 'bottom'
            else:
                label_y = height - err - 5  # Below the error bar
                va = 'top'
            ax.annotate(f'{height:+.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, label_y),
                       xytext=(0, 0),
                       textcoords="offset points",
                       ha='center', va=va,
                       fontsize=12, fontweight='bold',
                       color=HEADING_COLOR)
    
    add_labels(bars1, jit_only_err)
    add_labels(bars2, jit_micro_err)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='#666666', linewidth=1, linestyle='-', zorder=1)
    
    # Title and labels (matching figure7/figure9 style)
    ax.text(0.5, 1.05, 'Estimated Improvement vs Traditional Approach', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            color=HEADING_COLOR, ha='center', va='bottom')
    
    ax.set_ylabel('Improvement (%)', fontweight='semibold', color=HEADING_COLOR, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color='#444444')
    ax.tick_params(colors='#555555')
    
    # Set y limits with some padding
    ax.set_ylim(-130, 130)
    
    # Grid
    ax.grid(True, axis='y', linestyle=':', alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend (moved up slightly)
    legend = ax.legend(loc='upper right', frameon=False, fontsize=11, bbox_to_anchor=(1.0, 1.08))
    for text in legend.get_texts():
        text.set_color('#444444')
    
    # Add annotation box (right side, bigger)
    annotation_text = (
        "Key Findings:\n\n"
        "• Both approaches eliminate\n"
        "  wastage (100% improvement)\n\n"
        "• JIT-Only increases shortages\n"
        "  (-100% = worse than baseline)\n\n"
        "• JIT+Micro reduces shortages\n"
        "  by 84% vs Traditional\n\n"
        "• Overall efficiency gain:\n"
        "  95% for JIT+Micro approach"
    )
    
    props = dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#AAAAAA', alpha=1.0)
    ax.text(0.98, 0.08, annotation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, color='#333333',
            multialignment='left')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure6_estimated_savings_v2.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figure6_estimated_savings_v2.pdf'), bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: figure6_estimated_savings_v2.png/pdf")


if __name__ == "__main__":
    generate_savings_visualization()
