import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_species_summary(df, save_path: str = None):
    """
    Create summary plots from species stats and optionally save them.
    
    Args:
        df (pd.DataFrame): summary dataframe
        save_path (str, optional): directory where plots should be saved.
                                   If None, plots are just shown.
    """


    metrics = ["historic", "actual", "change"]
    plot_df = df[df["metric"].isin(metrics)]


    # Make sure save path exists
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 1. Lineplot of mean over generations
    plt.figure(figsize=(12,6))
    sns.lineplot(
        data=plot_df,
        x="generation",
        y="mean",
        hue="metric",
        style="subbranch_id",
        markers=True,
        dashes=False
    )
    plt.title("Mean species count over generations")
    plt.ylabel("Mean")
    plt.xlabel("Generation")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "lineplot_species_mean.png"))
        plt.close()
    else:
        plt.show()

    # 2. Boxplot of distributions by branch and metric
    plt.figure(figsize=(12,6))
    sns.boxplot(data=plot_df, x="subbranch_id", y="mean", hue="metric")
    plt.xticks(rotation=45)
    plt.title("Distribution of species mean by subbranch")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "boxplot_species_mean.png"))
        plt.close()
    else:
        plt.show()

    # 3. Heatmap of actual mean
    pivot_df = plot_df.pivot_table(
        index=["generation","subbranch_id"],
        columns="metric",
        values="mean"
    ).reset_index()

    plt.figure(figsize=(10,6))
    sns.heatmap(
        pivot_df.pivot(index="subbranch_id", columns="generation", values="actual"),
        annot=True, fmt=".1f", cmap="YlGnBu"
    )
    plt.title("Heatmap of actual species mean")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "heatmap_species_actual.png"))
        plt.close()
    else:
        plt.show()
