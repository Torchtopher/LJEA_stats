"""
LJEA Macroinvertebrate Data - Group Comparison Figures
======================================================
Generates multi-site comparison boxplots for each group of monitoring sites.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import textwrap

# =============================================================================
# Configuration & Styling
# =============================================================================

OUTPUT_DIR = Path("LJEA_Macro")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# NCBI Score threshold (lower is better; <4.05 is "Excellent")
EXCELLENT_THRESH = 4.05

# =============================================================================
# Site Groups Configuration
# =============================================================================

SITE_GROUPS = {
    "Group 1": {
        "sites": [
            "Mackey Creek - Burnette Rd",
            "Mill Creek - Andrew's Geyser",
            "Crooked Creek - McHone Dr",
            "Buck Creek - Greenlee Park",
        ],
        "palette": "Reds",
    },
    "Group 2": {
        "sites": [
            "North Fork Catawba River - North Cove School Rd",
            "North Fork Catawba River - American Thread Rd",
            "Armstrong Creek - Hwy 221"
        ],
        "palette": "Purples",
    },
    "Group 3": {
        "sites": [
            "White Creek - Above Route 126",
            "Linville River - Griffin Cottage",
            "Linville River - Blue Ridge Picnic Area",
        ],
        "palette": "Blues",
    },
}

# =============================================================================
# Data Setup
# =============================================================================

csv_text = """Name,Fall 2018,Summer 2019,Summer 2020,Winter 2021,Summer 2022,Summer 2023,Summer 2024,Summer 2025
White Creek - Above Route 126,1.82,,3.34,3.69,2.50,3.24,3.33,3.53
"Paddy's Creek - OMV Trail",3.08,,3.60,2.90,3.05,,2.67,3.41
North Fork Catawba River - American Thread Rd,3.27,,4.56,4.58,4.29,,,4.44
North Fork Catawba River - North Cove School Rd,3.10,,2.89,3.49,3.97,,,2.65
"Catawba River - Parker Padgett Rd",2.56,,4.41,,,2.92,,
Mill Creek - Old Fort,2.60,2.72,3.76,3.16,3.60,,3.16,2.98
Crooked Creek - McHone Dr,3.95,3.71,3.62,3.46,3.96,,,2.93
Linville River - Newland Highway,2.13,2.92,3.03,,2.47,3.26,,
"Linville River - Blue Ridge Picnic Area",3.34,3.45,3.29,,3.10,3.16,,
"Armstrong Creek - Hwy 221",1.50,,4.05,3.72,3.59,4.05,,3.07
Forsyth Creek - Plantation Drive,,,,,4.61,4.08,,3.98
Mackey Creek - Burnette Rd,,3.08,2.77,,,3.07,,2.59
Dale's Creek - Lake James Road,,,,,,,,2.70
"Crooked Creek - Bat Cave Road",,3.92,2.84,,,4.95,4.69,4.01
Catawba River - Old Fort Ball Park,,,2.62,,,3.33,3.42,2.21
Mill Creek - Andrew's Geyser,,2.81,2.80,,,2.10,,2.03
"Curtis Creek - Curtis Creek Rd",,2.31,3.48,,,2.33,,3.56
Buck Creek - Greenlee Park,,,5.09,3.44,,4.24,,
"Linville River - Griffin Cottage",,,4.82,3.09,,,,3.08"""

df = pd.read_csv(io.StringIO(csv_text))

# Melt to long format
df_melted = df.melt(id_vars='Name', var_name='Period', value_name='NCBI_Score').dropna(subset=['NCBI_Score'])


def prepare_group_data(sites):
    """Filter data for a specific group of sites."""
    return df_melted[df_melted['Name'].isin(sites)].copy()


# =============================================================================
# Plotting Logic
# =============================================================================

def plot_all_sites_boxplots(data):
    """Create multi-site comparison boxplot for ALL sites, ordered by average score."""

    # Calculate average score per site and sort (descending: worst on left, best on right)
    site_avgs = data.groupby('Name')['NCBI_Score'].mean().sort_values(ascending=False)
    ordered_sites = site_avgs.index.tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create boxplot with viridis gradient (dark=worst on left, bright=best on right)
    sns.boxplot(
        data=data,
        x='Name',
        y='NCBI_Score',
        order=ordered_sites,
        hue='Name',
        hue_order=ordered_sites,
        palette='viridis',
        width=0.6,
        legend=False,
        ax=ax
    )

    # Add median line for all data with legend
    overall_median = data['NCBI_Score'].median()
    median_line = ax.axhline(y=overall_median, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.legend([median_line], [f'Overall Median ({overall_median:.2f})'], loc='upper left')

    # Title and labels
    ax.set_title('Distribution of NCBI Scores (2018-2025)', fontweight='bold')
    ax.set_ylabel('NCBI Score (lower is better)')
    ax.set_xlabel('')

    # Wrap x-axis labels to 2 lines and rotate diagonally
    wrapped_labels = [textwrap.fill(site, width=20) for site in ordered_sites]
    ax.set_xticklabels(wrapped_labels, rotation=70, ha='right')

    # Grid
    ax.grid(True, axis='y', alpha=0.3)

    # Add sample sizes below each site name
    for i, site in enumerate(ordered_sites):
        n = len(data[data['Name'] == site])
        ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]), ha='center', va='top', rotation=70,
                    xytext=(0, -25), textcoords='offset points', fontsize=8, color='gray')

    # Footer text
    fig.text(
        0.5, -0.05,
        'Sites ordered by average NCBI score (Worst to Best). Period: 2018 to 2025',
        ha='center',
        fontsize=9,
        style='italic'
    )

    plt.tight_layout()

    # Save
    filename = "macro_boxplots_all_sites.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_group_boxplots(data, group_name, palette='viridis'):
    """Create multi-site comparison boxplot for a group, ordered by average score."""

    # Calculate average score per site and sort (descending: worst on left, best on right)
    site_avgs = data.groupby('Name')['NCBI_Score'].mean().sort_values(ascending=False)
    ordered_sites = site_avgs.index.tolist()

    # Create figure - width scales with number of sites
    n_sites = len(ordered_sites)
    fig_width = max(10, n_sites * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Create boxplot with gradient palette
    sns.boxplot(
        data=data,
        x='Name',
        y='NCBI_Score',
        order=ordered_sites,
        hue='Name',
        hue_order=ordered_sites,
        palette=palette,
        width=0.6,
        legend=False,
        ax=ax
    )

    # Add "Excellent" threshold line

    # Title and labels
    ax.set_title(f'Distribution of NCBI Scores - {group_name} (2018-2025)', fontweight='bold')
    ax.set_ylabel('NCBI Score')
    ax.set_xlabel('')

    # X-axis labels (no rotation, centered)
    ax.tick_params(axis='x', rotation=0)
    plt.setp(ax.get_xticklabels(), ha='center')

    # Grid
    ax.grid(True, axis='y', alpha=0.3)

    # Add sample sizes below each site name (using annotate with offset like original)
    for i, site in enumerate(ordered_sites):
        n = len(data[data['Name'] == site])
        ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]), ha='center', va='top',
                    xytext=(0, -25), textcoords='offset points', fontsize=8, color='gray')

    # Footer text
    fig.text(
        0.5, -0.05,
        f'Sites ordered by average NCBI score (Worst to Best). Period: 2018 to 2025',
        ha='center',
        fontsize=9,
        style='italic'
    )

    plt.tight_layout()

    # Save
    filename = f"macro_boxplots_{group_name.lower().replace(' ', '_')}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("LJEA Macroinvertebrate Data - Generating Group Figures")
    print("=" * 60)

    # Generate all-sites plot first
    print("\nGenerating all-sites comparison plot...")
    plot_all_sites_boxplots(df_melted)

    # Generate plots for each site group
    for group_name, group_config in SITE_GROUPS.items():
        print(f"\nGenerating plot for {group_name}...")
        sites = group_config["sites"]
        palette = group_config["palette"]
        group_data = prepare_group_data(sites)

        if len(group_data) > 0:
            plot_group_boxplots(group_data, group_name, palette)
        else:
            print(f"  Warning: No data found for {group_name}")

    print("\n" + "=" * 60)
    print(f"All macroinvertebrate figures saved to {OUTPUT_DIR}/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
