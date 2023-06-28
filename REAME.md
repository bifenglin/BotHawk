# BotHawk
##  A model for Bots detection in Open Source Software Projects
### 1. Introduction

This project is for detect bots in Open Source Software Projects.

- We collected a standardized dataset of 19,779 accounts to facilitate further studies on bots in open-source projects using systematic workflow to ensure data accuracy, generalization ability, scalability, and timeliness.
- We identify 17 distinguishing features that provide insight into bot behavior in open-source software
  projects.
- we introduce a novel bot detection model, BotHawk, which is capable of accurately identifying OSS bot in open-source software projects. We conducted a comparative analysis between our proposed model and state-of-the-art models. BotHawk
  exhibits strong performance in the AUC and F1-score indicators, achieving 0.947 and 0.89.


### 2. Project structure

- BotHawk data in `data/bothawk_data.csv`.

- BotHawk model in `model/BotHawk.pickle`

- Evaluating result and predict result in `result/*`

- [bothawk_model.py](bothawk_model.py) is the main file to train the BotHawk model.

- [bothunter_bench.py](bothunter_bench.py) is the main file to train the Bothunter model.

- [BoDeGHa_bench.py](BoDeGHa_bench.py) is the main file to train the BoDeGha model.

- [base_model_benmark.py](base_model_benmark.py) is the main file to evaluate the base model.

- `chart/*` is the files can draw the chart in the paper.

- [bothawk_feature.ipynb](bothawk_feature.ipynb) is the file to extract the features of BotHawk.
