
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

# Load data
df = pd.read_csv('data/LJEA_VWINdata.csv')

# Create proper datetime column
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Filter to 2018-current
df = df[df['year'] >= 2018].copy()

# Create site mapping: use most recent name for each site number
df_sorted = df.sort_values('date', ascending=False)
SITE_MAPPING = df_sorted.groupby('site #').first()['site name'].to_dict()

# Add canonical site name column based on site number
df['site'] = df['site #'].map(SITE_MAPPING)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSite # -> Name Mapping:")
for num in sorted([k for k in SITE_MAPPING.keys() if k != '-'], key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else x)):
    print(f"  {num}: {SITE_MAPPING[num]}")

# Chemical parameters (excluding metals which are 100% missing since 2018)
chem_cols = ['NH3-N', 'NO3-N', 'Total P', 'PO4', 'Turb', 'TSS', 'Cond', 'Alk', 'pH']

# Units and reference values from DATA_INFO.txt
# Note: Total P is measured as PO4; divide by 3.07 to get Total P as P
UNITS = {
    'NH3-N': 'mg/L',
    'NO3-N': 'mg/L',
    'Total P': 'mg/L as PO₄ (÷3.07 for P)',
    'PO4': 'mg/L',
    'Turb': 'NTU',
    'TSS': 'mg/L',
    'Cond': 'µmhos/cm',
    'Alk': 'mg/L',
    'pH': 's.u.',
}

# Reference values for comparison (2018-20 averages from DATA_INFO.txt)
REGIONAL_AVG = {
    'NH3-N': 0.08, 'NO3-N': 0.5, 'PO4': 0.08, 'Turb': 6.0,
    'TSS': 7.9, 'Cond': 77.7, 'Alk': 22.2, 'pH': 7.2,
}
PRISTINE_AVG = {
    'NH3-N': 0.04, 'NO3-N': 0.2, 'PO4': 0.04, 'Turb': 2.1,
    'TSS': 2.8, 'Cond': 19.7, 'Alk': 9.5, 'pH': 7.0,
}
NC_LIMITS = {
    'NO3-N': 10.0,  # nitrate only
    'Turb': 50.0,   # aquatic life (10 for Trout Waters)
    'pH': (6.0, 9.0),  # allowable range
}

print("\n--- Summary Statistics for Chemical Parameters ---")
print(df[chem_cols].describe())

# Missing data analysis
print("\n--- Missing Data (%) ---")
missing_pct = (df[chem_cols].isna().sum() / len(df) * 100).round(1)
print(missing_pct)

# Plot 1: Distribution of key parameters
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(chem_cols):
    ax = axes[i]
    data = df[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel(f'{col} ({UNITS[col]})')
        ax.set_ylabel('Count')

plt.suptitle('Distribution of Chemical Parameters (2018-present)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Time series of key nutrients
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

key_params = ['NH3-N', 'NO3-N', 'Total P', 'pH']
# Get top 6 sites by count using canonical site name
top_sites = df['site'].value_counts().head(6).index.tolist()

for ax, param in zip(axes.flatten(), key_params):
    for site in top_sites:
        site_data = df[df['site'] == site].sort_values('date')
        ax.plot(site_data['date'], site_data[param], 'o-', alpha=0.5, markersize=3, label=site)
    ax.set_title(param)
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{param} ({UNITS[param]})')
    ax.axvline(pd.Timestamp('2024-09-27'), color='red', linestyle='--', alpha=0.7, label='Hurricane Helene')

axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.suptitle('Key Parameters Over Time by Site (2018-present)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Boxplots by site
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, param in zip(axes.flatten(), key_params):
    site_order = df.groupby('site')[param].median().sort_values().index
    sns.boxplot(data=df, x='site', y=param, ax=ax, order=site_order)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_fontsize(8)
    ax.set_title(param)
    ax.set_xlabel('Site')
    ax.set_ylabel(f'{param} ({UNITS[param]})')

plt.suptitle('Parameter Distribution by Site (2018-present)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_boxplots_by_site.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 4: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[chem_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix of Chemical Parameters', fontsize=14, fontweight='bold')
ax.set_xlabel('Parameter')
ax.set_ylabel('Parameter')
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- EDA plots saved ---")
