"""
LJEA Lake Sites Correlation Analysis
=====================================
Creates correlation matrix heatmaps for NH3-N, Total P, and Chlorophyll a
at Lake James monitoring sites.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("LJEA")
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

LAKE_SITES = {
    '6': 'Plantation Point',
    '7': 'Big Island',
    '8': 'Marion Lake Club',
    '9': 'Paddy Creek Dam',
    '11': 'Lower Linville',
}

# Variable labels for display
VAR_LABELS = {
    'NH3-N': 'NH₃-N (mg/L)',
    'Total P': 'Total P (mg/L as PO₄)',
    'Chlor_a': 'Chlorophyll a (µg/L)',
}


def load_chemical_data():
    """Load chemical data from CSV, filter to lake sites and 2018+."""
    df = pd.read_csv('data/LJEA_VWINdata.csv')

    # Create date column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Convert site # to string for consistent matching
    df['site'] = df['site #'].astype(str)

    # Filter to lake sites (6, 7, 8, 9, 11) and years 2018+
    lake_site_nums = list(LAKE_SITES.keys())
    df = df[df['site'].isin(lake_site_nums)]
    df = df[df['year'] >= 2018]

    # Keep relevant columns
    df = df[['site', 'year', 'month', 'date', 'NH3-N', 'Total P']].copy()

    return df


def load_chlorophyll_data():
    """Load chlorophyll data from Excel file, combining all years."""
    years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    all_data = []

    for year in years:
        try:
            # Read the sheet for this year
            df = pd.read_excel('data/LJEA_chloro_data.xlsx', sheet_name=year, header=None)

            # Data starts at row 5 (0-indexed), with columns:
            # Column 1: Date, 2: LJ 6, 3: LJ 7, 4: LJ 8, 5: LJ 9, 6: LJ 11
            # (Column 0 is empty, columns 7+ are means)
            df = df.iloc[5:, [1, 2, 3, 4, 5, 6]].copy()
            df.columns = ['date', '6', '7', '8', '9', '11']

            # Keep only rows with actual date data (not annual mean row)
            df = df.dropna(subset=['date'])
            df = df[df['date'].apply(lambda x: not isinstance(x, str) or 'mean' not in str(x).lower())]

            # Parse dates
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            # Melt to long format
            df_long = df.melt(
                id_vars=['date'],
                value_vars=['6', '7', '8', '9', '11'],
                var_name='site',
                value_name='Chlor_a'
            )
            df_long['year'] = df_long['date'].dt.year
            df_long['month'] = df_long['date'].dt.month

            all_data.append(df_long)
        except Exception as e:
            print(f"Warning: Could not load chlorophyll data for {year}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def merge_data(chem_df, chlor_df):
    """Merge chemical and chlorophyll data by site, year, and month."""
    # Merge on site, year, and month (since exact dates may differ)
    merged = pd.merge(
        chem_df,
        chlor_df[['site', 'year', 'month', 'Chlor_a']],
        on=['site', 'year', 'month'],
        how='inner'
    )

    # Keep only rows with all three values
    merged = merged.dropna(subset=['NH3-N', 'Total P', 'Chlor_a'])

    return merged


def plot_correlation_all_sites(merged_df):
    """Create correlation matrix heatmap for all lake sites combined."""
    # Select only the variables of interest
    corr_vars = ['NH3-N', 'Total P', 'Chlor_a']
    df_corr = merged_df[corr_vars].copy()

    # Calculate correlation matrix
    corr_matrix = df_corr.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create heatmap with annotations
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax,
        xticklabels=[VAR_LABELS[v] for v in corr_vars],
        yticklabels=[VAR_LABELS[v] for v in corr_vars],
    )

    n_samples = len(df_corr)
    ax.set_title(f'Correlation Matrix - All Lake Sites Combined\n(n = {n_samples} samples)')

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'lake_correlation_all_sites.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return corr_matrix


def plot_correlation_per_site(merged_df):
    """Create individual correlation matrices for each lake site."""
    corr_vars = ['NH3-N', 'Total P', 'Chlor_a']

    for site_num, site_name in LAKE_SITES.items():
        # Filter to this site
        site_df = merged_df[merged_df['site'] == site_num][corr_vars].copy()

        if len(site_df) < 3:
            print(f"Warning: Site {site_num} ({site_name}) has only {len(site_df)} samples, skipping.")
            continue

        # Calculate correlation matrix
        corr_matrix = site_df.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))

        # Create heatmap with annotations
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax,
            xticklabels=[VAR_LABELS[v] for v in corr_vars],
            yticklabels=[VAR_LABELS[v] for v in corr_vars],
        )

        n_samples = len(site_df)
        ax.set_title(f'Correlation Matrix - Site {site_num}: {site_name}\n(n = {n_samples} samples)')

        plt.tight_layout()

        output_path = OUTPUT_DIR / f'lake_correlation_site_{site_num}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")


def main():
    """Main function to generate correlation matrices."""
    print("Loading chemical data...")
    chem_df = load_chemical_data()
    print(f"  Loaded {len(chem_df)} chemical records from lake sites (2018+)")

    print("Loading chlorophyll data...")
    chlor_df = load_chlorophyll_data()
    print(f"  Loaded {len(chlor_df)} chlorophyll records")

    print("Merging datasets...")
    merged_df = merge_data(chem_df, chlor_df)
    print(f"  Merged dataset has {len(merged_df)} samples with all 3 variables")

    if len(merged_df) == 0:
        print("Error: No samples with all three variables. Check date matching.")
        return

    print("\nGenerating correlation plots...")
    print("\n--- All Sites Combined ---")
    corr_all = plot_correlation_all_sites(merged_df)
    print(corr_all)

    print("\n--- Per-Site Matrices ---")
    plot_correlation_per_site(merged_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
