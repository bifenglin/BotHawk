import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../result/BotHawk_perm_imp.csv')

features = data['Feature']
importance = data['Feature Importance']

plt.figure(figsize=(14, 6))  # 设置图表尺寸
sns.barplot(x=importance, y=features)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.yticks(rotation=30)

plt.show()

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("../data/bothawk_data.csv")

X = dataframe.iloc[:, 1:-1]  # 特征数据
y = dataframe.iloc[:, -1]  # 标签数据

X = np.clip(X, 0, None)

selector = SelectKBest(chi2, k=5)
selector.fit(X, y)

scores = selector.scores_
feature_names = list(X.columns.values)
result_df = pd.DataFrame({"feature_names": feature_names, "scores": scores})
# result_df = result_df.sort_values(by="scores", ascending=False)

print(result_df)


plt.figure(figsize=(14, 6))
sns.barplot(x="scores", y="feature_names", data=result_df)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Scores')
plt.ylabel('Features')
plt.title('Feature Importance Evaluation using Chi-square Test')
plt.yticks(rotation=30)
plt.show()
