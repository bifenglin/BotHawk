import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../result/model_evaluation_v1.csv')

sns.set(style='whitegrid')

fig, axs = plt.subplots(ncols=4, sharey=True, figsize=(16,4))

sns.barplot(x='Accuracy', y='Model', data=df, ax=axs[0])
sns.barplot(x='Precision', y='Model', data=df, ax=axs[1])
sns.barplot(x='Recall', y='Model', data=df, ax=axs[2])
sns.barplot(x='F1-score', y='Model', data=df, ax=axs[3])

fig.suptitle('Comparison of Classification Models', fontsize=16)
axs[0].set_ylabel('')
axs[0].set_xlabel('Accuracy')
axs[1].set_xlabel('Precision')
axs[2].set_xlabel('Recall')
axs[3].set_xlabel('F1-score')

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


plt.tight_layout()
plt.show()
