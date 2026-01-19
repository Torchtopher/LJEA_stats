"""
LJEA Chemical Data Graphs - Production Quality Figures
======================================================
Generates paper-ready figures for Lake James water quality chemical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Output directory
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

# Color palette - colorblind friendly
SITE_COLORS = {
    '2': '#0077BB',   # Blue
    '5': '#33BBEE',   # Cyan
    '13': '#009988',  # Teal
    '17': '#EE7733',  # Orange
    '19': '#CC3311',  # Red
}

LAKE_COLORS = {
    '6': '#0077BB',   # Blue
    '7': '#33BBEE',   # Cyan
    '8': '#009988',   # Teal
    '9': '#EE7733',   # Orange
    '11': '#CC3311',  # Red
}

# Site mappings
RIVER_SITES = {
    '2': 'Catawba River @ 221A',
    '5': 'Linville River @ Hwy 126',
    '13': 'North Fork below Baxter',
    '17': 'North Fork @ Old North Cove School',
    '19': 'Armstrong Creek',
}

LAKE_SITES = {
    '6': 'Plantation Point',
    '7': 'Big Island',
    '8': 'Marion Lake Club',
    '9': 'Paddy Creek Dam',
    '11': 'Lower Linville',
}

# Units from DATA_INFO.txt
UNITS = {
    'NH3-N': 'mg/L',
    'PO4': 'mg/L',
    'TSS': 'mg/L',
    'Total P': 'mg/L as PO₄',
    'Chlor_a': 'µg/L',
}

# Reference values from DATA_INFO.txt (2018-20 averages)
REGIONAL_AVG = {'NH3-N': 0.08, 'PO4': 0.08, 'TSS': 7.9}
PRISTINE_AVG = {'NH3-N': 0.04, 'PO4': 0.04, 'TSS': 2.8}

# =============================================================================
# Data Loading
# =============================================================================

def load_chemical_data():
    """Load and prepare chemical data from CSV."""
    df = pd.read_csv('data/LJEA_VWINdata.csv')
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['site #'] = df['site #'].astype(str)
    df = df[df['year'] >= 2018].copy()
    return df


def load_chlorophyll_data():
    """Load and combine chlorophyll data from all yearly sheets."""
    xl = pd.ExcelFile('data/LJEA_chloro_data.xlsx')

    all_data = []
    for year in range(2018, 2026):
        sheet_name = str(year)
        if sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
            # Data starts at row 5 (0-indexed: 4), columns 1-6
            for row_idx in range(5, len(df)):
                date_val = df.iloc[row_idx, 1]
                if pd.isna(date_val) or date_val == 'mean' or date_val == 'Mean':
                    continue
                for col_idx, site in enumerate(['6', '7', '8', '9', '11'], start=2):
                    val = df.iloc[row_idx, col_idx]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        # Parse date
                        if isinstance(date_val, str):
                            try:
                                date = pd.to_datetime(f"{date_val} {year}")
                            except:
                                continue
                        else:
                            date = pd.to_datetime(date_val)
                        all_data.append({
                            'date': date,
                            'site #': site,
                            'Chlor_a': val
                        })

    return pd.DataFrame(all_data)


# =============================================================================
# Time Series Plots
# =============================================================================

# def plot_time_series_po4_nh3(df):
#     """
#     Create time series plots for PO4 and NH3-N at river sites.
#     Two-panel figure with shared x-axis.
#     """
#     fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     params = ['PO4', 'NH3-N']
#     titles = ['Orthophosphate (PO₄)', 'Ammonia-Nitrogen (NH₃-N)']

#     for ax, param, title in zip(axes, params, titles):
#         for site_id, site_name in RIVER_SITES.items():
#             site_data = df[df['site #'] == site_id].sort_values('date')
#             if len(site_data) > 0:
#                 ax.plot(site_data['date'], site_data[param],
#                        'o-', color=SITE_COLORS[site_id],
#                        label=f"Site {site_id}: {site_name}",
#                        markersize=4, linewidth=1, alpha=0.8)

#                 # Add reference lines
#             if param in REGIONAL_AVG:
#                 # Draw lines
#                 y_reg = REGIONAL_AVG[param]
#                 y_pri = PRISTINE_AVG[param]

#                 ax.axhline(y=y_reg, color='gray', linestyle='--', linewidth=2, alpha=1.0)
#                 ax.axhline(y=y_pri, color='green', linestyle=':', linewidth=2, alpha=1.0)

#                 # After plotting, get current x-limits
#                 x0, x1 = ax.get_xlim()
#                 pad = 0.01 * (x1 - x0)

#                 # Add text labels aligned left
#                 ax.text(x0 + pad, y_reg,
#                         f"Regional Avg ({y_reg})",
#                         ha='left', va='center')

#                 ax.text(x0 + pad, y_pri,
#                         f"Pristine Avg ({y_pri})",
#                         ha='left', va='center')
#         # Hurricane Helene marker
#         ax.axvline(x=pd.Timestamp('2024-09-27'), color='red', linestyle='--',
#                   linewidth=1.5, alpha=0.7)

#         ax.set_ylabel(f'{param} ({UNITS[param]})')
#         ax.set_title(title, fontweight='bold')
#         ax.grid(True, alpha=0.3)

#     # Add Hurricane Helene annotation to top plot
#     axes[0].annotate('Hurricane\nHelene', xy=(pd.Timestamp('2024-09-27'), axes[0].get_ylim()[1]),
#                     xytext=(pd.Timestamp('2024-06-01'), axes[0].get_ylim()[1]*0.85),
#                     fontsize=8, ha='center',
#                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

#     # Format x-axis
#     axes[1].xaxis.set_major_locator(mdates.YearLocator())
#     axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#     axes[1].set_xlabel('Date')

#     # Legend - moved to left side to avoid covering Hurricane Helene annotation
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.02, 0.5),
#               ncol=1, frameon=True, fancybox=True)

#     fig.suptitle('Chemical Parameters at River Monitoring Sites (2018-2025)',
#                 fontweight='bold', y=1.02)

#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / 'timeseries_PO4_NH3_river_sites.png',
#                bbox_inches='tight', facecolor='white')
#     plt.close()
#     print(f"Saved: {OUTPUT_DIR / 'timeseries_PO4_NH3_river_sites.png'}")


def plot_time_series_individual(df, param, sites_dict, colors_dict, site_type='River'):
    """Create individual time series plot for a single parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for site_id, site_name in sites_dict.items():
        site_data = df[df['site #'] == site_id].sort_values('date')
        if len(site_data) > 0 and param in site_data.columns:
            data = site_data.dropna(subset=[param])
            if len(data) > 0:
                ax.plot(data['date'], data[param],
                       'o-', color=colors_dict[site_id],
                       label=f"Site {site_id}: {site_name}",
                       markersize=4, linewidth=1, alpha=0.8)

        # Reference lines for river sites
        if site_type == 'River' and param in REGIONAL_AVG:
            # draw the lines
            y_reg = REGIONAL_AVG[param]
            y_pri = PRISTINE_AVG[param]

            ax.axhline(y=y_reg, color='gray', linestyle='--', linewidth=2, alpha=1.0)
            ax.axhline(y=y_pri, color='green', linestyle=':', linewidth=2, alpha=1.0)

            # find left edge of x-axis in data coords
            x0, x1 = ax.get_xlim()
            pad = 0.01 * (x1 - x0)   # small horizontal padding

            # textual labels positioned at left, vertically centered on the line
            ax.text(x0 + pad, y_reg,
                    f"Regional Avg ({y_reg})",
                    ha='left', va='center')

            ax.text(x0 + pad, y_pri,
                    f"Pristine Avg ({y_pri})",
                    ha='left', va='center')


    # Hurricane Helene marker
    ax.axvline(x=pd.Timestamp('2024-09-27'), color='red', linestyle='--',
              linewidth=1.5, alpha=0.7, label='Hurricane Helene')

    ax.set_ylabel(f'{param} ({UNITS.get(param, "")})')
    ax.set_xlabel('Date')
    ax.set_title(f'{param} at {site_type} Sites (2018-2025)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.legend(loc='upper left', bbox_to_anchor=(-0.25, 1), frameon=True)

    plt.tight_layout()
    filename = f'timeseries_{param}_{site_type.lower()}_sites.png'
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


# =============================================================================
# Box Plots
# =============================================================================

def plot_boxplots_river_sites(df):
    """Create box plots for PO4, NH3-N, TSS at river sites."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    params = ['PO4', 'NH3-N', 'TSS']
    titles = ['Orthophosphate (PO₄)', 'Ammonia-Nitrogen (NH₃-N)', 'Total Suspended Solids (TSS)']

    # Prepare data
    river_df = df[df['site #'].isin(RIVER_SITES.keys())].copy()
    river_df['Site'] = river_df['site #'].map(lambda x: f"{x}: {RIVER_SITES[x][:15]}...")

    for ax, param, title in zip(axes, params, titles):
        # Get data for this parameter
        plot_data = river_df.dropna(subset=[param])

        # Remove highest 2 outliers for TSS
        if param == 'TSS' and len(plot_data) > 2:
            sorted_tss = plot_data[param].sort_values(ascending=False)
            threshold = sorted_tss.iloc[2]  # Keep up to third highest
            plot_data = plot_data[plot_data[param] <= threshold]

        if len(plot_data) > 0:
            # Order by site number
            site_order = [f"{s}: {RIVER_SITES[s][:15]}..." for s in ['2', '5', '13', '17', '19']]
            site_order = [s for s in site_order if s in plot_data['Site'].unique()]

            sns.boxplot(data=plot_data, x='Site', y=param, ax=ax,
                       order=site_order, hue='Site', palette='Set2',
                       width=0.6, legend=False)

            # Add reference lines with improved visibility
            if param in REGIONAL_AVG:
                ax.axhline(y=REGIONAL_AVG[param], color='gray', linestyle='--',
                          linewidth=2, alpha=1.0)
                ax.axhline(y=PRISTINE_AVG[param], color='green', linestyle=':',
                          linewidth=2, alpha=1.0)
                # Add labels to reference lines
                ax.text(ax.get_xlim()[1], REGIONAL_AVG[param], f'Regional Avg ({REGIONAL_AVG[param]})', 
                       fontsize=8, va='bottom', ha='right', color='gray', backgroundcolor='white')
                ax.text(ax.get_xlim()[1], PRISTINE_AVG[param], f'Pristine Avg ({PRISTINE_AVG[param]})', 
                       fontsize=8, va='bottom', ha='right', color='green', backgroundcolor='white')

            # Add sample sizes
            for i, site in enumerate(site_order):
                n = len(plot_data[plot_data['Site'] == site])
                ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]),
                           ha='center', va='top', fontsize=8, color='gray')

            # Add note for TSS outliers
            if param == 'TSS':
                ax.text(0.02, 0.98, 'Note: Highest 2 TSS outliers removed', 
                       transform=ax.transAxes, fontsize=8, va='top', ha='left', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_ylabel(f'{param} ({UNITS[param]})')
        ax.set_xlabel('')
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, axis='y', alpha=0.3)

    # Add period of record note with better positioning
    date_range = f"{river_df['date'].min().strftime('%Y-%m')} to {river_df['date'].max().strftime('%Y-%m')}"
    fig.text(0.5, -0.08, f'Period of Record: {date_range}', ha='center', fontsize=9, style='italic')

    fig.suptitle('Chemical Parameters at River Monitoring Sites', fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boxplots_river_sites.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'boxplots_river_sites.png'}")


# def plot_boxplots_lake_sites(df, chlor_df):
#     """Create box plots for NH3-N, Total P, and Chlorophyll a at lake sites."""
#     fig, axes = plt.subplots(1, 3, figsize=(14, 5))

#     # Prepare lake data
#     lake_df = df[df['site #'].isin(LAKE_SITES.keys())].copy()
#     lake_df['Site'] = lake_df['site #'].map(lambda x: f"{x}: {LAKE_SITES[x]}")

#     chlor_df_plot = chlor_df.copy()
#     chlor_df_plot['Site'] = chlor_df_plot['site #'].map(lambda x: f"{x}: {LAKE_SITES[x]}")

#     site_order = [f"{s}: {LAKE_SITES[s]}" for s in ['6', '7', '8', '9', '11']]

#     # Plot 1: NH3-N
#     ax = axes[0]
#     plot_data = lake_df.dropna(subset=['NH3-N'])
#     sns.boxplot(data=plot_data, x='Site', y='NH3-N', ax=ax,
#                order=site_order, hue='Site', palette='Blues', width=0.6, legend=False)
#     ax.set_ylabel(f'NH₃-N ({UNITS["NH3-N"]})')
#     ax.set_xlabel('')
#     ax.set_title('Ammonia-Nitrogen (NH₃-N)', fontweight='bold')
#     ax.tick_params(axis='x', rotation=0)
#     ax.grid(True, axis='y', alpha=0.3)
#     for i, site in enumerate(site_order):
#         n = len(plot_data[plot_data['Site'] == site])
#         ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]), ha='center', va='top', fontsize=8, color='gray')

#     # Plot 2: Total P
#     ax = axes[1]
#     plot_data = lake_df.dropna(subset=['Total P'])
#     sns.boxplot(data=plot_data, x='Site', y='Total P', ax=ax,
#                order=site_order, hue='Site', palette='Greens', width=0.6, legend=False)
#     ax.set_ylabel(f'Total P ({UNITS["Total P"]})')
#     ax.set_xlabel('')
#     ax.set_title('Total Phosphorus (as PO₄)', fontweight='bold')
#     ax.tick_params(axis='x', rotation=0)
#     ax.grid(True, axis='y', alpha=0.3)
#     for i, site in enumerate(site_order):
#         n = len(plot_data[plot_data['Site'] == site])
#         ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]), ha='center', va='top', fontsize=8, color='gray')

#     # Plot 3: Chlorophyll a
#     ax = axes[2]
#     sns.boxplot(data=chlor_df_plot, x='Site', y='Chlor_a', ax=ax,
#                order=site_order, hue='Site', palette='YlGn', width=0.6, legend=False)
#     ax.set_ylabel(f'Chlorophyll a ({UNITS["Chlor_a"]})')
#     ax.set_xlabel('')
#     ax.set_title('Chlorophyll a', fontweight='bold')
#     ax.tick_params(axis='x', rotation=0)
#     ax.grid(True, axis='y', alpha=0.3)

#     # Chlorophyll thresholds
#     ax.axhline(y=2.5, color='green', linestyle=':', linewidth=1, alpha=0.7)
#     ax.axhline(y=8, color='orange', linestyle=':', linewidth=1, alpha=0.7)
#     ax.annotate('Oligotrophic', xy=(ax.get_xlim()[1], 2.5), fontsize=7, color='green', va='bottom')
#     ax.annotate('Eutrophic', xy=(ax.get_xlim()[1], 8), fontsize=7, color='orange', va='bottom')

#     for i, site in enumerate(site_order):
#         n = len(chlor_df_plot[chlor_df_plot['Site'] == site])
#         ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]), ha='center', va='top', fontsize=8, color='gray')

#     # Add period of record note
#     chem_range = f"Chemical: {lake_df['date'].min().strftime('%Y-%m')} to {lake_df['date'].max().strftime('%Y-%m')}"
#     chlor_range = f"Chlorophyll: {chlor_df['date'].min().strftime('%Y-%m')} to {chlor_df['date'].max().strftime('%Y-%m')}"
#     fig.text(0.5, -0.02, f'Period of Record — {chem_range}; {chlor_range}',
#             ha='center', fontsize=9, style='italic')

#     fig.suptitle('Chemical Parameters at Lake James Sites', fontweight='bold', y=1.02)

#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / 'boxplots_lake_sites.png', bbox_inches='tight', facecolor='white')
#     plt.close()
#     print(f"Saved: {OUTPUT_DIR / 'boxplots_lake_sites.png'}")


# =============================================================================
# Individual Site Time Series (one graph per site)
# =============================================================================

def plot_time_series_by_site(df):
    start_date = pd.Timestamp('2017-12-01')
    end_date = pd.Timestamp('2026-2-1')
    """Create individual time series plots for each site showing PO4 and NH3-N."""
    for site_id, site_name in RIVER_SITES.items():
        site_data = df[df['site #'] == site_id].sort_values('date')

        if len(site_data) == 0:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        params = ['PO4', 'NH3-N']
        titles = ['Orthophosphate (PO₄)', 'Ammonia-Nitrogen (NH₃-N)']
        colors = ['#0077BB', '#EE7733']

        for ax, param, title, color in zip(axes, params, titles, colors):
            data = site_data.dropna(subset=[param])
            if len(data) > 0:
                ax.plot(data['date'], data[param], 'o-', color=color,
                       markersize=5, linewidth=1.5, alpha=0.8)

                # Add reference lines
                if param in REGIONAL_AVG:
                    ax.axhline(y=REGIONAL_AVG[param], color='gray', linestyle='--',
                              linewidth=2, alpha=1.0, label=f'Regional Avg ({REGIONAL_AVG[param]})')
                    ax.axhline(y=PRISTINE_AVG[param], color='green', linestyle=':',
                              linewidth=2, alpha=1.0, label=f'Pristine Avg ({PRISTINE_AVG[param]})')

                # Hurricane Helene marker
                ax.axvline(x=pd.Timestamp('2024-09-27'), color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label='Hurricane Helene')

                ax.set_ylabel(f'{param} ({UNITS[param]})')
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left', fontsize=8)
                ax.set_xlim(start_date, end_date)


        # Format x-axis
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[1].set_xlabel('Date')

        fig.suptitle(f'Site {site_id}: {site_name}\nChemical Parameters (2018-2025)',
                    fontweight='bold', y=1.02)

        plt.tight_layout()
        filename = f'timeseries_site_{site_id}.png'
        plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / filename}")


# =============================================================================
# Individual Parameter Box Plots (one graph per parameter)
# =============================================================================

def plot_boxplot_single_param_river(df, param, title, palette='Set2'):
    """Create a single box plot for one parameter across river sites."""
    fig, ax = plt.subplots(figsize=(12, 7))

    river_df = df[df['site #'].isin(RIVER_SITES.keys())].copy()
    river_df['Site'] = river_df['site #'].map(lambda x: f"{x}: {RIVER_SITES[x]}")

    plot_data = river_df.dropna(subset=[param])

    # Remove highest 2 outliers for TSS
    if param == 'TSS' and len(plot_data) > 2:
        sorted_tss = plot_data[param].sort_values(ascending=False)
        threshold = sorted_tss.iloc[6]  # Keep up to fifth highest
        plot_data = plot_data[plot_data[param] <= threshold]

    if len(plot_data) == 0:
        plt.close()
        return

    site_order = [f"{s}: {RIVER_SITES[s]}" for s in ['2', '5', '13', '17', '19']]
    site_order = [s for s in site_order if s in plot_data['Site'].unique()]

    sns.boxplot(data=plot_data, x='Site', y=param, ax=ax,
               order=site_order, hue='Site', palette=palette, width=0.6, legend=False)

    # Add reference lines with improved visibility
    if param in REGIONAL_AVG:
        line_regional = ax.axhline(
            y=REGIONAL_AVG[param], color='gray', linestyle='--', linewidth=2, alpha=1.0
        )
        line_pristine = ax.axhline(
            y=PRISTINE_AVG[param], color='green', linestyle=':', linewidth=2, alpha=1.0
        )

        # Add labels to reference lines (annotations)
        # ax.text(ax.get_xlim()[1], REGIONAL_AVG[param], 
        #         f'Regional Avg ({REGIONAL_AVG[param]})', 
        #         fontsize=9, va='bottom', ha='right', color='gray', backgroundcolor='white')
        # ax.text(ax.get_xlim()[1], PRISTINE_AVG[param], 
        #         f'Pristine Avg ({PRISTINE_AVG[param]})', 
        #         fontsize=9, va='bottom', ha='right', color='green', backgroundcolor='white')

        # Add to legend
        ax.legend(handles=[line_regional, line_pristine],
                labels=[f'Regional Avg ({REGIONAL_AVG[param]})', 
                        f'Pristine Avg ({PRISTINE_AVG[param]})'],
                loc='upper right')

    # Add sample sizes
    for i, site in enumerate(site_order):
        n = len(plot_data[plot_data['Site'] == site])
        ax.text(i, -0.08, f'n={n}', ha='center', va='top', fontsize=9, color='gray', transform=ax.get_xaxis_transform())

    # Add note for TSS outliers
    if param == 'TSS':
        ax.text(0.02, 0.98, 'Note: Highest 6 TSS outliers removed', 
               transform=ax.transAxes, fontsize=9, va='top', ha='left', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_ylabel(f'{param} ({UNITS[param]})')
    ax.set_xlabel('Site')
    ax.set_title(f'{title}\nRiver Monitoring Sites (2018-2025)', fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, axis='y', alpha=0.3)

    # Add period of record with better positioning
    date_range = f"{river_df['date'].min().strftime('%Y-%m')} to {river_df['date'].max().strftime('%Y-%m')}"
    ax.annotate(f'Period of Record: {date_range}', xy=(0.5, -0.15), xycoords='axes fraction',
               ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    filename = f'boxplot_river_{param.replace("-", "").replace(" ", "_")}.png'
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_boxplot_single_param_lake(df, chlor_df, param, title, palette='Blues'):
    """Create a single box plot for one parameter across lake sites."""
    fig, ax = plt.subplots(figsize=(10, 6))

    site_order = [f"{s}: {LAKE_SITES[s]}" for s in ['6', '7', '8', '9', '11']]

    if param == 'Chlor_a':
        plot_df = chlor_df.copy()
        plot_df['Site'] = plot_df['site #'].map(lambda x: f"{x}: {LAKE_SITES[x]}")
        data_col = 'Chlor_a'
    else:
        lake_df = df[df['site #'].isin(LAKE_SITES.keys())].copy()
        lake_df['Site'] = lake_df['site #'].map(lambda x: f"{x}: {LAKE_SITES[x]}")
        plot_df = lake_df.dropna(subset=[param])
        data_col = param

    if len(plot_df) == 0:
        plt.close()
        return

    sns.boxplot(data=plot_df, x='Site', y=data_col, ax=ax,
               order=site_order, hue='Site', palette=palette, width=0.6, legend=False)

    # Add thresholds for chlorophyll
    if param == 'Chlor_a':
        line_oligo = ax.axhline(y=2.5, color='green', linestyle=':', linewidth=2, alpha=1.0)
        line_eutro = ax.axhline(y=8, color='orange', linestyle=':', linewidth=2, alpha=1.0)

        # Add text annotations (optional)
        # ax.annotate('Oligotrophic (<2.5)', xy=(ax.get_xlim()[1], 2.5), fontsize=8, color='green',
        #             va='bottom', ha='right')
        # ax.annotate('Eutrophic (>8)', xy=(ax.get_xlim()[1], 8), fontsize=8, color='orange',
        #             va='bottom', ha='right')

        # ➜ Add legend entries
        ax.legend(handles=[line_oligo, line_eutro],
                labels=['Oligotrophic (<2.5)', 'Eutrophic (>8)'],
                loc='upper right')
    # Add sample sizes
    for i, site in enumerate(site_order):
        n = len(plot_df[plot_df['Site'] == site])
        ax.text(i, -0.10, f'n={n}', ha='center', va='top', fontsize=9, color='gray', transform=ax.get_xaxis_transform())

    ax.set_ylabel(f'{param} ({UNITS[param]})')
    ax.set_xlabel('Site')
    ax.set_title(f'{title}\nLake James Sites (2018-2025)', fontweight='bold')
    ax.tick_params(axis='x', rotation=0) # prev 45
    ax.grid(True, axis='y', alpha=0.3)

    # Add period of record
    date_range = f"{plot_df['date'].min().strftime('%Y-%m')} to {plot_df['date'].max().strftime('%Y-%m')}"
    ax.annotate(f'Period of Record: {date_range}', xy=(0.5, -0.18), xycoords='axes fraction',
               ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    filename = f'boxplot_lake_{param.replace("-", "").replace(" ", "_")}.png'
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("LJEA Chemical Data - Generating Production Figures")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_chemical_data()
    print(f"  Chemical data: {len(df)} records (2018+)")

    chlor_df = load_chlorophyll_data()
    print(f"  Chlorophyll data: {len(chlor_df)} records")

    # Generate combined time series plots (all sites overlaid)
    # print("\nGenerating combined time series plots...")
    # plot_time_series_po4_nh3(df)

    # Individual time series for each parameter (all sites overlaid)
    for param in ['PO4', 'NH3-N']:
        plot_time_series_individual(df, param, RIVER_SITES, SITE_COLORS, 'River')

    # Individual time series for each site (PO4 and NH3 per site)
    print("\nGenerating individual site time series...")
    plot_time_series_by_site(df)

    # Generate combined box plots (3 params in one figure)
    print("\nGenerating combined box plots...")
    plot_boxplots_river_sites(df)
    #plot_boxplots_lake_sites(df, chlor_df)

    # Generate individual parameter box plots - River sites
    print("\nGenerating individual parameter box plots (river)...")
    plot_boxplot_single_param_river(df, 'PO4', 'Orthophosphate (PO₄)', palette='Blues')
    plot_boxplot_single_param_river(df, 'NH3-N', 'Ammonia-Nitrogen (NH₃-N)', palette='Oranges')
    plot_boxplot_single_param_river(df, 'TSS', 'Total Suspended Solids (TSS)', palette='Greens')

    # Generate individual parameter box plots - Lake sites
    print("\nGenerating individual parameter box plots (lake)...")
    plot_boxplot_single_param_lake(df, chlor_df, 'NH3-N', 'Ammonia-Nitrogen (NH₃-N)', palette='Blues')
    plot_boxplot_single_param_lake(df, chlor_df, 'Total P', 'Total Phosphorus (as PO₄)', palette='Greens')
    plot_boxplot_single_param_lake(df, chlor_df, 'Chlor_a', 'Chlorophyll a', palette='YlGn')

    print("\n" + "=" * 60)
    print("All chemical data figures saved to LJEA/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
