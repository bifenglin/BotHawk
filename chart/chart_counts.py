import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
BASE_DIR = '/Users/bifenglin/Code/maxcompute/scripts/bothawk/'

df = pd.read_csv(BASE_DIR + 'data/bothawk_data.csv')
df = df[df['label'].notnull()]

df["label"] = pd.Categorical(df["label"], categories=["Bot", "Human"], ordered=True)

bot_data = df[df['label'] == 'Bot']
human_data = df[df['label'] == 'Human']
variables = ['Number of Activity', 'Number of Issue', 'Number of Pull Request', 'Number of Repository',
             'Number of Commit', 'Number of Active day']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, var in enumerate(variables):
    row = i // 3
    col = i % 3
    sns.violinplot(ax=axs[row][col], x='label', y=var, data=df, inner='box', scale='width', hue='label',
                   hue_order=["Bot", "Human"], palette={"Bot": "blue", "Human": "orange"})
    axs[row][col].set_title(var.capitalize() + ' Distribution', fontsize=16)
    axs[row][col].set_xlabel('Label', fontsize=14)
    axs[row][col].set_ylabel(var.capitalize(), fontsize=14)
    axs[row][col].tick_params(axis='both', which='major', labelsize=12)
    if var not in ['activity_day']:
        axs[row][col].set_yscale('log')
    axs[row][col].legend(title="Label", fontsize=12, title_fontsize=14, loc="upper left",
                         bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

plt.tight_layout()

sns.set(style="ticks", font_scale=1.2, rc={
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,
    'axes.edgecolor': 'k',
    'xtick.major.size': 4,
    'xtick.major.width': 1.2,
    'ytick.major.size': 4,
    'ytick.major.width': 1.2,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.edgecolor': 'k',
    'legend.framealpha': 0.9,
    'legend.shadow': True,
    'axes.grid': True,
    'grid.color': 'gray',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'figure.autolayout': True})

plt.show()
