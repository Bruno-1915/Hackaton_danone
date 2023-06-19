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

warnings.filterwarnings('ignore')
# %%
cols = ['is_beverage', 'non_recyclable_and_non_biodegradable_materials_count', 
        'est_co2_agriculture', 'est_co2_consumption', 
        'est_co2_distribution', 'est_co2_packaging', 'est_co2_processing', 
        'est_co2_transportation'] 
target_col = 'ecoscore_grade'

# %%
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
dtrain = xgb.DMatrix(X_train, label=y_train.values)
dvalid = xgb.DMatrix(X_test)
# %%
evals_result = {}
watchlist = [(dtrain, "train")]
params = {
    "seed": 0,
    "eta": 0.1,
    "alpha": 15,
    # "gamma": 25,
    # "lambda": 15,
    "max_depth": 15,
    "refresh_leaf": 1,
    "booster": "gbtree",
    "max_delta_step": 1,
    "n_estimators": 1000,
    # "colsample_bytree": 0.1,
    # "colsample_bylevel": 0.8,
    # "min_child_weight": 11.0,
    # "seed_per_iteration": True,
    'subsample': .1,
    "objective": "multi:softprob",
    'num_class': len(y_train.unique()),
    "eval_metric": ["auc", "mlogloss"],
}
eta_decay = np.linspace(params['eta'], 0.01, 2000).tolist()
# %%
model = xgb.train(
    params,
    dtrain,
    params.pop("n_estimators"),
    evals=watchlist,
    verbose_eval=True,
    evals_result=evals_result,
    callbacks=[xgb.callback.LearningRateScheduler(eta_decay)],
)
# %%
Path(product['result'])\
    .write_text(json.dumps(
        {'target': {index: int(i) 
                    for index, i in enumerate(np.argmax(model.predict(dvalid), axis=1))}
         }))
