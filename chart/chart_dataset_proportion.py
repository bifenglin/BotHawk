import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = '/Users/bifenglin/Code/maxcompute/scripts/bothawk/'
bothawk_clean_data = pd.read_csv('data/bothawk_clean_data_v2.csv')
bothawk_active_account = pd.read_csv('../data/bothawk_active_account.csv')
bothawk_random_account = pd.read_csv('../data/bothawk_random_account.csv')
BIMAN_account = pd.read_csv('../data/BIMAN_account.csv')
BoDeGHa_account = pd.read_csv('../data/BoDeGHa_account.csv')

def compute_actor_id_proportion(dataset):
    actor_ids = set(dataset['actor_id'])
    common_actor_ids = actor_ids.intersection(set(bothawk_clean_data['actor_id']))
    proportion = len(common_actor_ids) / len(bothawk_clean_data['actor_id'])
    return proportion

proportions = {}
proportions['bothawk_active_account'] = compute_actor_id_proportion(bothawk_active_account)
proportions['bothawk_random_account'] = compute_actor_id_proportion(bothawk_random_account)
proportions['BIMAN_account'] = compute_actor_id_proportion(BIMAN_account)
proportions['BoDeGHa_account'] = compute_actor_id_proportion(BoDeGHa_account)

for dataset, proportion in proportions.items():
    print(f"The proportion of actor_id in {dataset}.csv that appear in bothawk_clean_data.csv is {proportion:.2%}.")

fig, ax = plt.subplots()
datasets = list(proportions.keys())
proportions_values = list(proportions.values())
ax.pie(proportions_values, labels=datasets, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
ax.axis('equal')
# ax.set_title('Each dataset proportion in Bothawk', fontsize=16)
plt.tight_layout()

plt.show()

plt.savefig(BASE_DIR + 'chart/Each dataset proportion in Bothawk.png', dpi=600, bbox_inches='tight')
