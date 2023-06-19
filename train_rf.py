# %% tags=["parameters"]
upstream = None
product = None
# %%
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
# %%
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
# %%
cols = ['is_beverage', 'non_recyclable_and_non_biodegradable_materials_count', 
        'est_co2_agriculture', 'est_co2_consumption', 
        'est_co2_distribution', 'est_co2_packaging', 'est_co2_processing', 
        'est_co2_transportation']
target_col = 'ecoscore_grade'
# %%
print(cols)
# %%
# train = pd.read_csv(upstream['Preprocess features']['train_csv'])
train = pd.read_csv(upstream['Preprocess features']['train_csv'], usecols=cols + [target_col])
test = pd.read_csv(upstream['Preprocess features']['test_csv'], usecols=cols)

# %%
X_train, y_train = train[cols], train[target_col]
X_test = test[cols]
for i in cols:
    X_train[i] = X_train[i].astype(float)
    X_test[i] = X_test[i].astype(float)
# %%
print(X_train.shape, y_train.shape)
print(X_test.shape)
# %%
model = RandomForestClassifier(n_estimators=500, random_state=0)
# %%
model.fit(X_train, y_train)
# %%
Path(product['result']).write_text(json.dumps({'target': {index: int(i) for index, i in enumerate(model.predict(X_test))}}))