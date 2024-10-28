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

with open('config/cluster_dudep.pkl','rb') as f :
    kmeans = pickle.load(f)

best_k = 10

total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])

with open('config/xgb_params_info.pkl','rb') as f :
    params_list = pickle.load(f)

X_total.drop(['latitude','longitude'],axis=1,inplace=True)

xgb_models = []

for i in range(best_k):
    print(f'Cluster {i} modeling...')
    total_cluster_idx = np.where(total_pred == i)[0]   # (index_array, dtype)
    X_total_cluster = X_total.iloc[total_cluster_idx]
    y_total_cluster = y_total.iloc[total_cluster_idx]
    xgb_model = xgboost.XGBRegressor(**params_list[i])
    xgb_model.fit(X_total_cluster, y_total_cluster, verbose=20)
    xgb_models.append(xgb_model)

X_test['pred'] = 0
X_test = X_test.drop(columns=['latitude', 'longitude'])

for i in range(best_k):
    test_cluster_idx = np.where(test_pred == i)[0]
    X_test_cluster = X_test.iloc[test_cluster_idx]
    X_test.loc[X_test_cluster.index, 'pred'] = xgb_models[i].predict(X_test_cluster.drop(columns=['pred']))
    
test_pred_xgb_cluster = X_test['pred'] * X_test['area_m2']
sample_submission['deposit'] = test_pred_xgb_cluster
sample_submission.to_csv('results/final_output.csv', index=False, encoding='utf-8-sig')

print('End')