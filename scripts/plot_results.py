import os
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# USER DEFINED ACTION NAMES
# ============================================================
# Keys MUST match action indices produced by argmax(Q)
ACTION_NAMES = {
    0: "swap+2opt",
    1: "swap+LK",
    2: "rev+2opt",
    3: "rev+LK",
    4: "rand+2opt",
    5: "near+2opt",
    6: "cheap+2opt",
    7: "near+LK",
}

STATE_NAMES = {
    0: "EXCELENT",
    1: "GOOD",
    2: "REGULAR",
    3: "POOR",
    4: "BETTER",
}

NUM_ACTIONS = len(ACTION_NAMES)
NUM_STATES = len(STATE_NAMES)
UNVISITED_ACTION = NUM_ACTIONS  # special index


def experiments_plots():
    # ==========================================
    # 1. SETUP & DATA LOADING
    # ==========================================

    custom_palette = {
        "Random": "#333333",  # Dark Grey/Black
        "Base": "#d62728",  # Standard Red
        "Base (gamma 0.95)": "#ff9896",  # Light Red
        "Base (gamma 0.995)": "#9467bd",  # Darker Red/Purple tone or distinct
        # Actually let's keep gammas in Red family:
        "Base (gamma 0.995)": "#8c0000",  # Dark Red
        "Time Discount": "#2ca02c",  # Green
        "Trained 50 ATT": "#1f77b4",  # Medium Blue
        "Trained 50 GEO": "#aec7e8",  # Light Blue
        "Trained 50 EUC": "#000080",  # Navy Blue
        "Trained 50 ATT (Time)": "#9467bd",  # Medium Purple
        "Trained 50 GEO (Time)": "#c5b0d5",  # Light Purple
        "Trained 50 EUC (Time)": "#4b0082",  # Dark Indigo
    }

    # Define a logical order for the legend
    logical_order = [
        "Random",
        "Base",
        "Base (gamma 0.95)",
        "Base (gamma 0.995)",
        "Time Discount",
        "Trained 50 ATT",
        "Trained 50 GEO",
        "Trained 50 EUC",
        "Trained 50 ATT (Time)",
        "Trained 50 GEO (Time)",
        "Trained 50 EUC (Time)",
    ]

    # Define your experiments here: Key = Label for plot, Value = Filename
    # Update these paths to match your actual file locations
    experiments_files = {
        "Base": "data/results/resultsBase099.csv",
        "Base (gamma 0.95)": "data/results/resultsBase095.csv",
        "Base (gamma 0.995)": "data/results/resultsBase0995.csv",
        "Time Discount": "data/results/resultsTime099.csv",
        "Random": "data/results/resultsRandom.csv",
        "Trained 50 ATT": "data/results/resultsCrossATT099.csv",
        "Trained 50 GEO": "data/results/resultsCrossGEO099.csv",
        "Trained 50 EUC": "data/results/resultsCrossEUC_2D099.csv",
        "Trained 50 ATT (Time)": "data/results/resultsTimeCrossATT099.csv",
        "Trained 50 GEO (Time)": "data/results/resultsTimeCrossGEO099.csv",
        "Trained 50 EUC (Time)": "data/results/resultsTimeCrossEUC_2D099.csv",
    }

    output_dir = "data/plots"

    dfs = []

    # Load loop
    for label, file_path in experiments_files.items():
        # Check if file exists to avoid crashing
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Experiment"] = label  # Add a label column

            # Remove '%' from Gap and convert to float
            if "Gap" in df.columns:
                df["Gap"] = df["Gap"].astype(str).str.replace("%", "", regex=False)
                df["Gap"] = pd.to_numeric(df["Gap"], errors="coerce")

            # Ensure numeric types
            cols_to_numeric = ["Gap", "Time (ms)", "Dimension", "Total Iterations"]
            for col in cols_to_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

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
    plt.rcParams.update({"figure.dpi": 150})  # High res for papers

    # ==========================================
    # 2. VISUALIZATIONS
    # ==========================================

    # --- PLOT A: Scalability (Gap vs Dimension) ---
    # Goal: See if "Trained on 50" generalizes to 100
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=master_df,
        x="Dimension",
        y="Gap",
        hue="Experiment",
        style="Experiment",
        palette=custom_palette,  # <--- APPLYING COLORS
        hue_order=logical_order,  # <--- SORTING LEGEND
        style_order=logical_order,  # <--- SORTING STYLES
        markers=True,
        dashes=False,
        errorbar=None,
    )
    plt.title("Gap evolution by Problem Size", fontsize=14)
    plt.ylabel("Gap to BFS (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1A_scalability_gap_vs_dim_gap.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=master_df,
        x="Dimension",
        y="Time (ms)",
        hue="Experiment",
        style="Experiment",
        palette=custom_palette,  # <--- APPLYING COLORS
        hue_order=logical_order,  # <--- SORTING LEGEND
        style_order=logical_order,  # <--- SORTING STYLES
        markers=True,
        dashes=False,
        errorbar=None,
    )
    plt.title("Time evolution by Problem Size", fontsize=14)
    plt.ylabel("Average Time (ms)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1B_scalability_gap_vs_dim_time.png"))
    plt.close()

    # --- PLOT B: Efficiency Frontier (Time vs Gap) ---
    # Goal: Identify the "sweet spot" (bottom-left corner)
    # We aggregate by Experiment to see the overall trade-off
    summary_df = (
        master_df.groupby("Experiment")[["Gap", "Time (ms)"]].mean().reset_index()
    )

    plt.figure(figsize=(10, 8))
    p1 = sns.scatterplot(
        data=summary_df,
        x="Time (ms)",
        y="Gap",
        hue="Experiment",
        s=200,  # Marker size
        palette=custom_palette,
        edgecolor="black",
    )

    plt.title("Average Gap vs. Average Time", fontsize=14)
    plt.xlabel("Average Time (ms)")
    plt.ylabel("Average Gap (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_efficiency_time_vs_gap.png"))
    plt.close()

    # --- PLOT C: Topology Robustness (Gap by Type) ---

    # We use catplot to create a grid of subplots (facets)
    g = sns.catplot(
        data=master_df[master_df["Type"].isin(["EUC_2D", "GEO"])],
        x="Experiment",
        y="Gap",
        col="Type",  # <--- This splits the graph into 3 columns
        kind="violin",  # You can change this to 'box' if you prefer
        col_wrap=1,  # <--- Stacks them vertically (1 column) for better width
        aspect=2.5,  # Makes each plot wide enough to read labels
        height=4,  # Height of each row
        palette=custom_palette,
        hue="Experiment",
        order=logical_order,  # Ensures the order on X-axis is logical
        cut=0,  # Don't extend violin past data range
        sharex=False,  # Keeps x-labels independent if needed
    )

    # Adjust the Y-axis limit for all plots to zoom in on the important part
    # (Calculated based on 95th percentile to ignore extreme outliers)
    if not master_df["Gap"].isnull().all():
        y_lim = master_df["Gap"].quantile(0.95) * 1.5
        g.set(ylim=(0, y_lim))

    # Rotate the x-axis labels so they don't overlap
    g.set_xticklabels(rotation=45, ha="right")

    # Add titles and labels
    g.figure.subplots_adjust(top=0.92)  # Make room for the main title
    g.figure.suptitle("Gap Distribution by Problem Type", fontsize=16)
    g.set_axis_labels("", "Gap to BFS (%)")  # Remove X label (redundant)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3A_robustness_type_vs_gap.png"))
    plt.close()

    # We use catplot to create a grid of subplots (facets)
    g = sns.catplot(
        data=master_df[master_df["Type"].isin(["EUC_2D", "GEO"])],
        x="Experiment",
        y="Time (ms)",
        col="Type",  # <--- This splits the graph into 3 columns
        kind="violin",  # You can change this to 'box' if you prefer
        col_wrap=1,  # <--- Stacks them vertically (1 column) for better width
        aspect=2.5,  # Makes each plot wide enough to read labels
        height=4,  # Height of each row
        palette=custom_palette,
        hue="Experiment",
        order=logical_order,  # Ensures the order on X-axis is logical
        cut=0,  # Don't extend violin past data range
        sharex=False,  # Keeps x-labels independent if needed
    )

    # Adjust the Y-axis limit for all plots to zoom in on the important part
    # (Calculated based on 95th percentile to ignore extreme outliers)
    if not master_df["Time (ms)"].isnull().all():
        y_lim = master_df["Time (ms)"].quantile(0.95) * 1.5
        g.set(ylim=(0, y_lim))

    # Rotate the x-axis labels so they don't overlap
    g.set_xticklabels(rotation=45, ha="right")

    # Add titles and labels
    g.figure.subplots_adjust(top=0.92)  # Make room for the main title
    g.figure.suptitle("Time Distribution by Problem Type", fontsize=16)
    g.set_axis_labels("", "Average Time (ms)")  # Remove X label (redundant)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3B_robustness_type_vs_time.png"))
    plt.close()

    # --- PLOT D: Convergence Speed (Iterations vs Dimension) ---
    # Goal: See if Time Discount or Gamma changes convergence speed
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=master_df[
            master_df["Experiment"].isin(["Base", "Random", "Time Discount"])
        ],
        x="Dimension",
        y="Total Iterations",
        hue="Experiment",
        palette=custom_palette,
    )
    plt.title("Total Iterations by Problem Size", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_convergence_iterations.png"))
    plt.close()

    # ==========================================
    # 1. PREPARE NORMALIZED DATA
    # ==========================================

    # Filter only the relevant experiments
    # We need 'Base' for the math, and 'Trained 50' variants for the plot
    target_experiments = [
        "Base",
        "Trained 50 ATT",
        "Trained 50 GEO",
        "Trained 50 EUC",
        "Trained 50 ATT (Time)",
        "Trained 50 GEO (Time)",
        "Trained 50 EUC (Time)",
    ]
    subset_df = master_df[master_df["Experiment"].isin(target_experiments)].copy()

    # Aggregate by Experiment and Dimension first
    # We compare the average performance at each dimension
    agg_df = (
        subset_df.groupby(["Experiment", "Dimension"])[["Gap", "Time (ms)"]]
        .mean()
        .reset_index()
    )

    # Extract the 'Base' values to use as the denominator
    base_values = agg_df[agg_df["Experiment"] == "Base"][
        ["Dimension", "Gap", "Time (ms)"]
    ]
    base_values = base_values.rename(
        columns={"Gap": "Base_Gap", "Time (ms)": "Base_Time"}
    )

    # Merge Base values back into the main dataframe
    merged_df = pd.merge(agg_df, base_values, on="Dimension", how="left")

    # Calculate Ratios
    merged_df["Norm_Gap"] = merged_df["Gap"] / merged_df["Base_Gap"]
    merged_df["Norm_Time"] = merged_df["Time (ms)"] / merged_df["Base_Time"]

    # Remove 'Base' rows from the final plot data (since they would just be 1.0, 1.0)
    plot_df = merged_df[merged_df["Experiment"] != "Base"]

    # ==========================================
    # 2. GENERATE FACETED PLOT
    # ==========================================

    # Using relplot to create a grid of scatter plots (one per Dimension)
    g = sns.relplot(
        data=plot_df,
        x="Norm_Time",
        y="Norm_Gap",
        hue="Experiment",
        col="Dimension",  # Create one subplot per Dimension
        col_wrap=4,  # 5 graphs per row (adjust based on screen size)
        palette=custom_palette,  # Use your custom colors
        hue_order=[e for e in logical_order if e in plot_df["Experiment"].unique()],
        s=150,  # Marker size
        edgecolor="black",
        alpha=0.8,
        height=3.5,
        aspect=1,
    )

    # ==========================================
    # 3. ADD REFERENCE LINES & FORMATTING
    # ==========================================

    # Add lines at 1.0 to show the "Base" performance threshold
    for ax in g.axes.flat:
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)

        # Optional: Shade the "Win" quadrant (Bottom-Left)
        # This highlights models that are strictly better than Base
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # We set a semi-transparent green box from 0,0 to 1,1
        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (0, 0), 1, 1, linewidth=0, edgecolor="none", facecolor="green", alpha=0.05
        )
        ax.add_patch(rect)

    # Titles
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle("Relative Performance vs. Base Model (Normalized)", fontsize=16)

    g.set_axis_labels("Normalized Time (x Base)", "Normalized Gap (x Base)")

    plt.savefig(os.path.join(output_dir, "5_relative_performance.png"))
    plt.close()


# ============================================================
# Q-table loading
# ============================================================
def load_qtable_txt(path):
    with open(path, "r") as f:
        num_states, num_actions = map(int, f.readline().split())
        if num_actions != NUM_ACTIONS:
            raise ValueError(
                f"{path}: expected {NUM_ACTIONS} actions, found {num_actions}"
            )
        data = [list(map(float, f.readline().split()))
                for _ in range(num_states)]
    return np.array(data)


def extract_policy(qtable):
    policy = np.empty(qtable.shape[0], dtype=int)
    for s in range(qtable.shape[0]):
        if np.all(qtable[s] == 0.0):
            policy[s] = UNVISITED_ACTION
        else:
            policy[s] = np.argmax(qtable[s])
    return policy


# ============================================================
# File handling
# ============================================================
def collect_txt_files(folder):
    files = []
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            if fname.endswith(".txt"):
                files.append(os.path.join(root, fname))
    return files


def extract_n(filename):
    match = re.match(r"instance_size_(\d+)\.txt", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1))


def load_model(folder):
    entries = []
    for path in collect_txt_files(folder):
        n = extract_n(os.path.basename(path))
        qtable = load_qtable_txt(path)
        policy = extract_policy(qtable)
        entries.append((n, policy))

    if not entries:
        raise RuntimeError(f"No Q-tables found in {folder}")

    entries.sort(key=lambda x: x[0])
    ns = [n for n, _ in entries]
    policy_matrix = np.stack([p for _, p in entries], axis=1)

    return ns, policy_matrix


# ============================================================
# Plotting
# ============================================================
def plot_models(model_data, labels, out_path):
    """
    model_data: list of (ns, policy_matrix)
    labels: list of model names
    """

    num_models = len(model_data)

    # ---------- consistency checks ----------
    ns_ref = model_data[0][0]
    num_states = model_data[0][1].shape[0]

    for ns, mat in model_data:
        if ns != ns_ref:
            raise ValueError("All models must share the same n values")
        if mat.shape[0] != num_states:
            raise ValueError("All models must share the same states")

    # ---------- colormap ----------
    colors_group_1 = plt.cm.Blues(np.linspace(0.4, 0.9, 4))
    colors_group_2 = plt.cm.Reds(np.linspace(0.4, 0.9, 4))
    color_unvisited = np.array([[0.7, 0.7, 0.7, 1.0]])

    colors = np.vstack([colors_group_1, colors_group_2, color_unvisited])

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, NUM_ACTIONS + 1.5, 1),
        ncolors=NUM_ACTIONS + 1,
    )

    # ---------- figure ----------
    fig, axes = plt.subplots(
        1, num_models,
        figsize=(4 * num_models + 2, 6),
        sharey=True,
        squeeze=False
    )
    axes = axes[0]

    fig.suptitle("Policy Heatmap", fontsize=14)

    for ax, (ns, mat), label in zip(axes, model_data, labels):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[-0.5, mat.shape[1] - 0.5, -0.5, mat.shape[0] - 0.5],
        )

        ax.set_title(label)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns, rotation=45)
        ax.set_xlabel("Instance size (n)")

    axes[0].set_ylabel("State")

    # ---------- single shared colorbar ----------
    cbar = fig.colorbar(
        im, ax=axes.tolist(), ticks=list(range(NUM_ACTIONS + 1))
    )
    cbar.ax.set_yticklabels(
        [ACTION_NAMES[a] for a in range(NUM_ACTIONS)] + ["Unvisited"]
    )
    cbar.set_label("Chosen action")

    axes[0].set_yticks(range(num_states))
    axes[0].set_yticklabels([STATE_NAMES[a] for a in range(NUM_STATES)] )

    fig.savefig(out_path, dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot policy heatmaps (1 or 3 models)"
    )
    parser.add_argument(
        "model_dirs",
        nargs="+",
        help="One or three model directories",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional model labels",
    )
    parser.add_argument(
        "--out",
        default="policy_heatmap.png",
        help="Output image filename",
    )

    args = parser.parse_args()

    if len(args.model_dirs) not in (1, 3):
        raise ValueError("Provide either 1 or 3 model directories")

    labels = args.labels
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(args.model_dirs))]
    elif len(labels) != len(args.model_dirs):
        raise ValueError("Number of labels must match number of models")

    model_data = [load_model(d) for d in args.model_dirs]

    plot_models(model_data, labels, args.out)


if __name__ == "__main__":
    main()

