import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# 1. Configuration
# =============================================================================

# Output directory
OUTPUT_DIR = Path("LJEA_habitat")
OUTPUT_DIR.mkdir(exist_ok=True)

# Figure settings for paper-ready output
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# 2. Site Groups Configuration
# =============================================================================

SITE_GROUPS = {
    "Group 1": {
        "sites": [
            "North Fork @ Old North Cove School Rd. (LJ17)",
            "North Fork - American Thread Rd.",
            "Armstrong Creek (LJ19)",
        ],
        "palette": "Reds",
    },
    "Group 2": {
        "sites": [
            "Linville @ BRP Picnic Area (LJ29)",
            "Linville River @ Griffin Cottage (LJ5*)",
        ],
        "palette": "Purples",
    },
    "Group 3": {
        "sites": [
            "Upper Mill @ Andrews Geyser",
            "Mill Creek in Old Fort (LJ25)",
            "Curtis Creek off Curtis Creek Rd",
            "Catawba @ Greenlee Park (LJ31)",
        ],
        "palette": "Blues",
    },
    "Group 4": {
        "sites": [
            "Paddy's Creek in State Park (LJ23)",
            "White Creek above Rt. 126 (LJ27)",
            "Forsyth Creek (LJ26)",
        ],
        "palette": "Greens",
    },
}

# =============================================================================
# 3. Data Loading & Prep
# =============================================================================

def load_habitat_data():
    """Load habitat data from CSV."""
    df = pd.read_csv("data/LJEA_habitat_cleaned.csv")
    return df


def prepare_group_data(df, sites):
    """Filter and prepare data for a specific group of sites."""
    # Filter for specific sites
    group_df = df.loc[df['Location'].isin(sites)].copy()

    # Define year columns
    year_cols = [str(y) for y in range(2018, 2026)]

    # Melt to long format
    long_df = group_df.melt(
        id_vars="Location",
        value_vars=year_cols,
        var_name="Year",
        value_name="Score"
    )

    # Remove empty data
    long_df = long_df.dropna(subset=["Score"])

    # Ensure the order of sites matches the list
    long_df['Location'] = pd.Categorical(long_df['Location'], categories=sites, ordered=True)

    return long_df


# =============================================================================
# 4. Plotting Logic
# =============================================================================

def plot_habitat_scores(data, group_name, palette='Blues'):
    """Create habitat score boxplot for a group of sites."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create Boxplot using gradient palette for consistency with chemical graphs
    sns.boxplot(
        data=data,
        x='Location',
        y='Score',
        hue='Location',
        palette=palette,
        width=0.6,
        legend=False,
        ax=ax
    )

    # ---------------------------------------------------------
    # Styling and Annotations
    # ---------------------------------------------------------

    # Title and Axis Labels
    ax.set_title(f'Habitat Scores - {group_name} (2018-2025)', fontweight='bold')
    ax.set_ylabel('Habitat Score (higher is better)')
    ax.set_xlabel('')

    # Grid styling
    ax.grid(True, axis='y', alpha=0.3)

    # Rotate x-axis labels for readability
    ax.tick_params(axis='x', rotation=0)

    # Add Sample Size (n=) Annotations at the bottom of the plot
    sorted_sites = data['Location'].cat.categories

    for i, site in enumerate(sorted_sites):
        n = len(data[data['Location'] == site])
        if n > 0:
            ax.text(i, -0.07, f'n={n}', ha='center', va='top', fontsize=9, color='gray', transform=ax.get_xaxis_transform())

    # Add "Period of Record" Footer
    years = data['Year'].unique()
    min_year = min(years)
    max_year = max(years)

    fig.text(
        0.5, -0.05,
        f'Period of Record: {min_year} to {max_year}',
        ha='center',
        fontsize=9,
        style='italic'
    )

    # Adjust layout
    plt.tight_layout()

    # Save plot (must be before plt.show() or figure will be cleared)
    filename = f"habitat_scores_{group_name.lower().replace(' ', '_')}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


# =============================================================================
# 5. Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("LJEA Habitat Data - Generating Production Figures")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_habitat_data()
    print(f"  Habitat data: {len(df)} sites")

    # Generate plots for each site group
    for group_name, group_config in SITE_GROUPS.items():
        print(f"\nGenerating plot for {group_name}...")
        sites = group_config["sites"]
        palette = group_config["palette"]
        group_data = prepare_group_data(df, sites)

        if len(group_data) > 0:
            plot_habitat_scores(group_data, group_name, palette)
        else:
            print(f"  Warning: No data found for {group_name}")

    print("\n" + "=" * 60)
    print(f"All habitat figures saved to {OUTPUT_DIR}/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
