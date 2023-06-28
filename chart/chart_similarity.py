import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('data/bothawk_clean_data_v2.csv')
df = df[df['label'].notnull()]
df["label"] = pd.Categorical(df["label"], categories=["Bot", "Human"], ordered=True)

bot_data = df[df['label'] == 'Bot']
human_data = df[df['label'] == 'Human']

fig, axs = plt.subplots(ncols=3, figsize=(18,6))

sns.violinplot(x='label', y='jaccard_similarity', data=df, inner='box', scale='width', hue='label',
               hue_order=["Bot", "Human"], palette={"Bot": "blue", "Human": "orange"}, ax=axs[0])
axs[0].set_title('Jaccard Similarity', fontsize=16)
axs[0].set_xlabel('Label', fontsize=14)
axs[0].set_ylabel('Jaccard Similarity', fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].legend(title="Label", fontsize=12, title_fontsize=14, loc="upper left",
              bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

# 绘制 TF-IDF 相似度的小提琴图
sns.violinplot(x='label', y='tfidf_similarity', data=df, inner='box', scale='width', hue='label',
               hue_order=["Bot", "Human"], palette={"Bot": "blue", "Human": "orange"}, ax=axs[1])
axs[1].set_title('TF-IDF Similarity', fontsize=16)
axs[1].set_xlabel('Label', fontsize=14)
axs[1].set_ylabel('TF-IDF Similarity', fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].legend(title="Label", fontsize=12, title_fontsize=14, loc="upper left",
              bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

sns.violinplot(x='label', y='cosin_similarity', data=df, inner='box', scale='width',hue='label',
               hue_order=["Bot", "Human"], palette={"Bot": "blue", "Human": "orange"}, ax=axs[2])
axs[2].set_title('Cosin Similarity', fontsize=16)
axs[2].set_xlabel('Label', fontsize=14)
axs[2].set_ylabel('Cosin Similarity', fontsize=14)
axs[2].tick_params(axis='both', which='major', labelsize=12)
axs[2].legend(title="Label", fontsize=12, title_fontsize=14, loc="upper left",
              bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)


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
