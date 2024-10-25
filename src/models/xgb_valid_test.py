import os
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.cluster import KMeans
import xgboost
from src.utils.HuberLoss import custom_loss,custom_metric

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_df = pd.read_csv('data/train_aftercountplace.csv',index_col=0)
test_df = pd.read_csv('data/test_aftercountplace.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

with open('config/index_list.pkl','rb') as f:
    index_list = pickle.load(f)

train_df = train_df.iloc[index_list,:]
train_df.reset_index(inplace=True,drop=True)
test_df.drop(axis=1,columns=['index'],inplace=True)
train_df = train_df.drop(columns=[ 'contract_day', 'age'])
test_df = test_df.drop(columns=['contract_day', 'age'])
train_df['contract_type'] = train_df['contract_type'].replace(2, np.nan)
test_df['contract_type'] = test_df['contract_type'].replace(2, np.nan)
train_df['deposit_per_area'] = train_df['deposit'] / train_df['area_m2']
train_df.drop(columns=['deposit'], inplace=True)
X_test = test_df.copy()
X_total = train_df.drop(columns=['deposit_per_area'])
y_total = train_df['deposit_per_area']

holdout_start = 202307
holdout_end = 202312
valid_df = train_df[(train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end)]
final_train_df = train_df[~((train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end))]

X_train = final_train_df.drop(columns=['deposit_per_area'])
y_train = final_train_df['deposit_per_area']
X_valid = valid_df.drop(columns=['deposit_per_area'])
y_valid = valid_df['deposit_per_area']
X_test = test_df.copy()

X_total = train_df.drop(columns=['deposit_per_area'])
y_total = train_df['deposit_per_area']


with open('config/cluster_dudep.pkl','rb') as f :
    kmeans = pickle.load(f)

best_k = 10

total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])
train_pred = kmeans.predict(X_train[['latitude', 'longitude']])
valid_pred = kmeans.predict(X_valid[['latitude', 'longitude']])

with open('config/xgb_params_info.pkl','rb') as f :
    params_list = pickle.load(f)

X_total.drop(['latitude','longitude'],axis=1,inplace=True)
X_train = X_train.drop(columns=['latitude', 'longitude'])
X_valid = X_valid.drop(columns=['latitude', 'longitude'])

xgb_models = []

print(X_train.columns)
print(len(X_train.columns))

print(X_valid.columns)
print(len(X_valid.columns))

print(X_test.columns)
print(len(X_test.columns))

for i in range(best_k):

    print(f'Cluster {i} modeling...')
    train_cluster_idx = np.where(train_pred == i)[0]   # (index_array, dtype)
    valid_cluster_idx = np.where(valid_pred == i)[0]

    X_train_cluster = X_train.iloc[train_cluster_idx]
    y_train_cluster = y_train.iloc[train_cluster_idx]

    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    y_valid_cluster = y_valid.iloc[valid_cluster_idx]




    xgb_model = xgboost.XGBRegressor(**params_list[i])
    xgb_model.fit(X_train_cluster, y_train_cluster, verbose=50)

    xgb_models.append(xgb_model)

X_valid['pred'] = 0
for i in range(best_k):
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    X_valid.loc[X_valid_cluster.index, 'pred'] = xgb_models[i].predict(X_valid_cluster.drop(columns=['pred']))


valid_mae = mean_absolute_error(y_valid * valid_df['area_m2'], X_valid['pred'] * X_valid['area_m2'])

print(valid_mae)
print('End')