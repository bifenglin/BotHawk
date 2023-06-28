import glob
import pandas as pd
import matplotlib.pyplot as plt

file_paths = glob.glob('./result/*_roc_curve_data.csv')

dfs = []
for file in file_paths:
    df = pd.read_csv(file)
    model_name = file.split('/')[-1].split('_roc')[0]
    df['model'] = model_name
    dfs.append(df)

fig, ax = plt.subplots(figsize=(8, 6))
for df in dfs:
    name = df.model.iloc[0]
    ax.plot(df['False Positive Rate'], df['True Positive Rate'], label=name)  # 绘制每个模型的 ROC 曲线  # 绘制每个模型的 ROC 曲线

ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier', color='grey')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

ax.legend()
plt.show()
