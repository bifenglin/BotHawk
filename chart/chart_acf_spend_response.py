import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
BASE_DIR = '/Users/bifenglin/Code/maxcompute/scripts/bothawk/'

df = pd.read_csv(BASE_DIR + 'data/bothawk_data.csv')
df = df[df['label'].notnull()]

df["label"] = pd.Categorical(df["label"], categories=["Bot", "Human"], ordered=True)

bot_data = df[df['label'] == 'Bot']
human_data = df[df['label'] == 'Human']

variables = ['connection_account', 'median_response_time', 'periodicity_of_activities']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

for i, var in enumerate(variables):
    if var == 'connection_account':
        var = 'counts_of_connection_account'
    sns.violinplot(ax=axs[i], x='label', y=var, data=df, inner='box', scale='width', hue='label',
                   hue_order=["Bot", "Human"], palette={"Bot": "blue", "Human": "orange"})
    if var == 'counts_of_connection_account':
        axs[i].set_title('Connection Account' + ' Distribution', fontsize=14, fontweight='bold')
    if var == 'median_response_time':
        axs[i].set_title('Median Response Time' + ' Distribution', fontsize=14, fontweight='bold')
    if var == 'periodicity_of_activities':
        axs[i].set_title('Periodicity of Activities' + ' Distribution', fontsize=14, fontweight='bold')
    axs[i].set_xlabel('Label', fontsize=14, fontweight='bold')
    axs[i].set_ylabel(var.capitalize(), fontsize=14, fontweight='bold')
    axs[i].tick_params(axis='both', which='major', labelsize=12)
    if var in ['median_response_time', 'counts_of_connection_account']:
        axs[i].set_yscale('log')
    axs[i].legend(title="Label", fontsize=12, title_fontsize=14, loc="upper left",
                  bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

plt.tight_layout()
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
