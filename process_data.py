# %% tags=["parameters"]
upstream = None
product = None
file_train_data = None
file_test_data = None

import warnings

# %%
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')

mlb = MultiLabelBinarizer()
# %%
os.makedirs('/'.join(product['train_csv'].split('/')[:-1]), exist_ok=True)

# %%
train = pd.read_json(file_train_data).T
test = pd.read_json(file_test_data).T
# %%
train['test_ingredients'] = train.ingredients.apply(lambda x: [i['id'] for i in x] if x != 'unknown' else ['unknown'])
test['test_ingredients'] = test.ingredients.apply(lambda x: [i['id'] for i in x] if x != 'unknown' else ['unknown'])
# %%
# temp = mlb.fit_transform(pd.concat([train.pop('test_ingredients'), test.pop('test_ingredients')]))
# train = train.join(pd.DataFrame(temp[:len(train),:],
#                           columns=mlb.classes_,
#                           index=train.index))

# test = test.join(pd.DataFrame(temp[len(train):,:],
#                           columns=mlb.classes_,
#                           index=test.index))
# # %%
# def fill_percentages(df):
#     for index, i in enumerate(df.ingredients):
#         if i == 'unknown':
#             df.unknown.iloc[index] = 100
#         else:
#             for ingredient in i:
#                 df[ingredient['id']].iloc[index] = ingredient['percent_estimate']
#     return df
            
# # %%
# train = fill_percentages(train)
# test = fill_percentages(test)
# %%
train.to_csv(product['train_csv'], index=False)
test.to_csv(product['test_csv'], index=False)
