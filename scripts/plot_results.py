import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================

custom_palette = {
    'Random':                 '#333333',  # Dark Grey/Black
    
    'Base':                   '#d62728',  # Standard Red
    'Base (gamma 0.95)':      '#ff9896',  # Light Red
    'Base (gamma 0.995)':     '#9467bd',  # Darker Red/Purple tone or distinct
    # Actually let's keep gammas in Red family:
    'Base (gamma 0.995)':     '#8c0000',  # Dark Red

    'Time Discount':          '#2ca02c',  # Green
    
    'Trained 50 ATT':         '#1f77b4',  # Medium Blue
    'Trained 50 GEO':         '#aec7e8',  # Light Blue
    'Trained 50 EUC':         '#000080',  # Navy Blue
    
    'Trained 50 ATT (Time)':  '#9467bd',  # Medium Purple
    'Trained 50 GEO (Time)':  '#c5b0d5',  # Light Purple
    'Trained 50 EUC (Time)':  '#4b0082',  # Dark Indigo
}

# Define a logical order for the legend
logical_order = [
    'Random',
    'Base', 'Base (gamma 0.95)', 'Base (gamma 0.995)',
    'Time Discount',
    'Trained 50 ATT', 'Trained 50 GEO', 'Trained 50 EUC',
    'Trained 50 ATT (Time)', 'Trained 50 GEO (Time)', 'Trained 50 EUC (Time)'
]

# Define your experiments here: Key = Label for plot, Value = Filename
# Update these paths to match your actual file locations
experiments_files = {
    'Base': 'data/results/resultsBase099.csv',
    'Base (gamma 0.95)': 'data/results/resultsBase095.csv',
    'Base (gamma 0.995)': 'data/results/resultsBase0995.csv',
    'Time Discount': 'data/results/resultsTime099.csv',
    'Random': 'data/results/resultsRandom.csv',
    'Trained 50 ATT': 'data/results/resultsCrossATT099.csv',
    'Trained 50 GEO': 'data/results/resultsCrossGEO099.csv',
    'Trained 50 EUC': 'data/results/resultsCrossEUC_2D099.csv',
    'Trained 50 ATT (Time)': 'data/results/resultsTimeCrossATT099.csv',
    'Trained 50 GEO (Time)': 'data/results/resultsTimeCrossGEO099.csv',
    'Trained 50 EUC (Time)': 'data/results/resultsTimeCrossEUC_2D099.csv',
}

output_dir = 'data/plots'

dfs = []

# Load loop
for label, file_path in experiments_files.items():
    # Check if file exists to avoid crashing
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Experiment'] = label # Add a label column
        
        # Remove '%' from Gap and convert to float
        if 'Gap' in df.columns:
            df['Gap'] = df['Gap'].astype(str).str.replace('%', '', regex=False)
            df['Gap'] = pd.to_numeric(df['Gap'], errors='coerce')

        # Ensure numeric types
        cols_to_numeric = ['Gap', 'Time (ms)', 'Dimension', 'Total Iterations']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        dfs.append(df)
    else:
        print(f"Warning: File not found {file_path}")

# Combine into one big DataFrame
if dfs:
    master_df = pd.concat(dfs, ignore_index=True)
else:
    raise ValueError("No data loaded. Please check file paths.")

# Set global aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 150}) # High res for papers

# ==========================================
# 2. VISUALIZATIONS
# ==========================================

# --- PLOT A: Scalability (Gap vs Dimension) ---
# Goal: See if "Trained on 50" generalizes to 100
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=master_df, 
    x='Dimension', 
    y='Gap', 
    hue='Experiment', 
    style='Experiment', 
    palette=custom_palette,      # <--- APPLYING COLORS
    hue_order=logical_order,     # <--- SORTING LEGEND
    style_order=logical_order,   # <--- SORTING STYLES
    markers=True, 
    dashes=False,
    errorbar=None 
)
plt.title('Gap evolution by Problem Size', fontsize=14)
plt.ylabel('Gap to Optimal (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_scalability_gap_vs_dim.png'))
plt.close()

# --- PLOT B: Efficiency Frontier (Time vs Gap) ---
# Goal: Identify the "sweet spot" (bottom-left corner)
# We aggregate by Experiment to see the overall trade-off
summary_df = master_df.groupby('Experiment')[['Gap', 'Time (ms)']].mean().reset_index()

plt.figure(figsize=(10, 8))
p1 = sns.scatterplot(
    data=summary_df, 
    x='Time (ms)', 
    y='Gap', 
    hue='Experiment', 
    s=200, # Marker size
    palette=custom_palette,
    edgecolor='black',

)

plt.title('Average Gap vs. Average Time', fontsize=14)
plt.xlabel('Average Time (ms)')
plt.ylabel('Average Gap (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_efficiency_time_vs_gap.png'))
plt.close()

# --- PLOT C: Topology Robustness (Gap by Type) ---
# --- PLOT C: Robustness (Separated by Type) ---

# We use catplot to create a grid of subplots (facets)
g = sns.catplot(
    data=master_df, 
    x='Experiment', 
    y='Gap', 
    col='Type',           # <--- This splits the graph into 3 columns
    kind='violin',        # You can change this to 'box' if you prefer
    col_wrap=1,           # <--- Stacks them vertically (1 column) for better width
    aspect=2.5,           # Makes each plot wide enough to read labels
    height=4,             # Height of each row
    palette=custom_palette, 
    hue='Experiment',
    order=logical_order,  # Ensures the order on X-axis is logical
    cut=0,                # Don't extend violin past data range
    sharex=False          # Keeps x-labels independent if needed
)

# Adjust the Y-axis limit for all plots to zoom in on the important part
# (Calculated based on 95th percentile to ignore extreme outliers)
if not master_df['Gap'].isnull().all():
    y_lim = master_df['Gap'].quantile(0.95) * 1.5
    g.set(ylim=(0, y_lim))

# Rotate the x-axis labels so they don't overlap
g.set_xticklabels(rotation=45, ha='right')

# Add titles and labels
g.fig.subplots_adjust(top=0.92) # Make room for the main title
g.fig.suptitle('Gap Distribution by Problem Type', fontsize=16)
g.set_axis_labels("", "Gap to Optimal (%)") # Remove X label (redundant)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_robustness_type_vs_gap.png'))
plt.close()

# --- PLOT D: Convergence Speed (Iterations vs Dimension) ---
# Goal: See if Time Discount or Gamma changes convergence speed
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=master_df[master_df["Experiment"].isin(["Base", "Random", "Time Discount"])],
    x='Dimension',
    y='Total Iterations',
    hue='Experiment',
    palette=custom_palette,
)
plt.title('Total Iterations by Problem Size', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_convergence_iterations.png'))
plt.close()