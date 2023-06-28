import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = '/Users/bifenglin/Code/maxcompute/scripts/bothawk/'

df = pd.read_csv(BASE_DIR + 'data/bothawk_data.csv')
df = df[df['label'].notnull()]

df["label"] = pd.Categorical(df["label"], categories=["Bot", "Human"], ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(ax=axes[0], x="label", y="Number of following", hue="label", data=df,
               palette={"Bot": "blue", "Human": "orange"})
axes[0].set_title("Number of Following Distribution (log scale)", fontsize=14)
axes[0].set_xlabel("Label", fontsize=12)
axes[0].set_ylabel("Number of Following (log scale)", fontsize=12)
axes[0].set_yscale('log')
axes[0].legend(title="Label", fontsize=10, title_fontsize=12, loc="upper left",
               bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

sns.violinplot(ax=axes[1], x="label", y="Number of followers", hue="label", data=df,
               palette={"Bot": "blue", "Human": "orange"})
axes[1].set_title("Number of Followers Distribution (log scale)", fontsize=14)
axes[1].set_xlabel("Label", fontsize=12)
axes[1].set_ylabel("Number of Followers (log scale)", fontsize=12)
axes[1].set_yscale('log')
axes[1].legend(title="Label", fontsize=10, title_fontsize=12, loc="upper left",
               bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

sns.set_style("ticks", {"axes.linewidth": 1.2, "axes.edgecolor": 'k',
                        "xtick.major.size": 4, "xtick.major.width": 1.2,
                        "ytick.major.size": 4, "ytick.major.width": 1.2,
                        "legend.frameon": True, "legend.edgecolor": 'k', "legend.framealpha": 0.9,
                        "legend.shadow": True, "axes.grid": True, "grid.color": 'gray',
                        "grid.linestyle": '-', "grid.linewidth": 0.5, "grid.alpha": 0.5})
sns.set_context("notebook", font_scale=1.2, rc={
    'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'figure.autolayout': True})

plt.tight_layout()
plt.show()

# plt.savefig(BASE_DIR + "chart/Following and Followers Distribution.jpg", dpi=300)
