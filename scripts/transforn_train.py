# going to export the notebook into a script here

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


import pickle

# get the data 
df_results = pd.read_csv('data/results.csv')
df_results = df_results[['raceId', 'driverId', 'constructorId', 'grid', 'position']]

df_constructors = pd.read_csv('data/constructors.csv')

df_drivers = pd.read_csv('data/drivers.csv')

df_races = pd.read_csv('data/races.csv')


# get podium rate n stuff
df_merge = df_results.merge(df_drivers[['driverId', 'driverRef']], on='driverId', how='left')
df_merge = df_merge.merge(df_constructors[['constructorId', 'constructorRef']], on='constructorId', how='left')
df_merge = df_merge.merge(df_races[['raceId', 'circuitId', 'year', 'round']], on='raceId', how='left')
df_merge


df_merge['podium'] = df_merge['position'].isin(['1', '2', '3'])
df_merge = df_merge.sort_values(['year', 'round'])

# filter to after 1980
df_merge = df_merge[df_merge['year'] > 1980]

# last year podium rate
def get_last_year_podium_rate(group):
    result = []
    for idx, row in group.iterrows():
        current_year = row['year']
        last_year_data = group[(group['year'] == current_year - 1)]
        if len(last_year_data) > 0:
            rate = last_year_data['podium'].mean()
        else:
            rate = np.nan
        result.append(rate)
    return pd.Series(result, index=group.index)

df_merge['podium_rate_last_year'] = df_merge.groupby('driverId').apply(get_last_year_podium_rate, include_groups=False).reset_index(level=0, drop=True)

# current year podium rate up to last race
def get_current_year_podium_rate(group):
    result = []
    for idx, row in group.iterrows():
        current_year = row['year']
        current_round = row['round']
        current_year_data = group[(group['year'] == current_year) & (group['round'] < current_round)]
        if len(current_year_data) > 0:
            rate = current_year_data['podium'].mean()
        else:
            rate = np.nan
        result.append(rate)
    return pd.Series(result, index=group.index)

df_merge['podium_rate_curr_year'] = df_merge.groupby('driverId').apply(get_current_year_podium_rate, include_groups=False).reset_index(level=0, drop=True)

# 3. all time podium rate til current race
df_merge['podium_rate_all_time'] = df_merge.groupby('driverId')['podium'].apply(
    lambda x: x.expanding().mean().shift(1)
).reset_index(level=0, drop=True)

# drop unnecessary columns
df_merge = df_merge.drop(columns=['raceId', 'driverId', 'constructorId', 'circuitId', 'position'])


# fillna
df_merge = df_merge.fillna(0) # default value to 0


# train test split
# split data into train validation and test



df_full_train, df_test = train_test_split(df_merge, test_size=0.2)
df_train, df_val = train_test_split(df_full_train, test_size=0.25)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = (df_train.podium).astype('int').values
y_val = (df_val.podium).astype('int').values
y_test = (df_test.podium).astype('int').values
y_full_train = (df_full_train.podium).astype('int').values

# Remove target variable from features to prevent leakage
del df_train['podium']

del df_val['podium']

del df_test['podium']

del df_full_train['podium']


#xgboost

# remade dv 
train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

# train model with good params
# now lets train the final model with the good params

# Get best params from grid search

xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# get full train
full_train_dicts = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(full_train_dicts)
features = list(dv.get_feature_names_out())
dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)

# Train final model on full training data
final_model = xgb.train(xgb_params, dfull_train, num_boost_round=200)




# export model
with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)
    

# i know I am supposed to make a function n stuff but i am incredibly lazy