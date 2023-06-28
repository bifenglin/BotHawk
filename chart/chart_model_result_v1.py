import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('result/model_evaluation_v1.csv')

# Set the plot style
sns.set(style='whitegrid')

# Create a subplot grid with 3 rows and 2 columns
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharey=True)

# Plot Accuracy in the first subplot
sns.barplot(x='Accuracy', y='Model', data=df, ax=axs[0, 0])

# Plot Precision in the second subplot
sns.barplot(x='Precision', y='Model', data=df, ax=axs[0, 1])

# Plot Recall in the third subplot
sns.barplot(x='Recall', y='Model', data=df, ax=axs[1, 0])

# Plot F1-score in the fourth subplot
sns.barplot(x='F1-score', y='Model', data=df, ax=axs[1, 1])

# Plot AUC in the fifth subplot, spanning both columns
sns.barplot(x='AUC', y='Model', data=df, ax=axs[2, :])

# Add a title and labels
fig.suptitle('Comparison of Classification Models', fontsize=16)
axs[0, 0].set_ylabel('')
axs[0, 1].set_ylabel('')
axs[0, 0].set_xlabel('Accuracy')
axs[0, 1].set_xlabel('Precision')
axs[1, 0].set_xlabel('Recall')
axs[1, 1].set_xlabel('F1-score')
axs[2, 0].set_xlabel('AUC')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
