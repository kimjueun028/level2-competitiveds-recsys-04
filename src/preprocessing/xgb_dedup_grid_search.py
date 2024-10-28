import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
import xgboost
from tqdm import tqdm
import pickle
from src.utils.HuberLoss import custom_loss, custom_metric
from sklearn.cluster import KMeans

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_df = pd.read_csv('data/train_aftercountplace.csv')
test_df = pd.read_csv('data/test_aftercountplace.csv')

with open('config/index_list.pkl','rb') as f:
    index_list = pickle.load(f)

train_df = train_df.iloc[index_list,:]
train_df.reset_index(inplace=True,drop=True)
test_df.drop('index',axis=1,inplace=True)

train_df = train_df.drop(columns=[ 'contract_day', 'age'])
test_df = test_df.drop(columns=['contract_day', 'age'])

train_df['contract_type'] = train_df['contract_type'].replace(2, np.nan)
test_df['contract_type'] = test_df['contract_type'].replace(2, np.nan)

train_df['deposit_per_area'] = train_df['deposit'] / train_df['area_m2']
train_df.drop(columns=['deposit'], inplace=True)

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

best_k = 10
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED)
kmeans.fit(X_total[['latitude','longitude']])

total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])

train_pred = kmeans.predict(X_train[['latitude', 'longitude']])
valid_pred = kmeans.predict(X_valid[['latitude', 'longitude']])
X_train = X_train.drop(columns=['latitude', 'longitude'])
X_valid = X_valid.drop(columns=['latitude', 'longitude'])

with open('cluster_dedep.pkl', 'wb') as f:
     pickle.dump(kmeans, f)

param_grid = {
    'objective': [custom_loss],   
    'eval_metric': [custom_metric],   
    'seed': [42],    
    'n_estimators': [300,500,1000,1500],
    'learning_rate': [0.01, 0.02],
    'max_depth': [6, 12],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'n_jobs' : [-1],
    'device' : ['cuda']
}

param_list = list(ParameterGrid(param_grid))
total_fits = len(param_list)

xgb_models = []
params_list = []

for i in range(best_k) :
    print(f'Cluster {i} modeling...')
    best_score = np.inf  # 최적의 점수 초기화
    best_params = None  # 최적의 파라미터 초기화
    best_model = None  # 최적의 모델 초기화
    train_cluster_idx = np.where(train_pred == i)[0]   # (index_array, dtype)
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_train_cluster = X_train.iloc[train_cluster_idx]
    y_train_cluster = y_train.iloc[train_cluster_idx]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    y_valid_cluster = y_valid.iloc[valid_cluster_idx]
    with tqdm(total=total_fits) as pbar:
     for params in param_list:
        xgb_model = xgboost.XGBRegressor(**params)
        xgb_model.fit(X_train_cluster, y_train_cluster, eval_set=[(X_valid_cluster, y_valid_cluster)], verbose=False)
        y_pred = xgb_model.predict(X_valid_cluster)
        score = mean_absolute_error(y_valid_cluster, y_pred)  # y_valid는 검증 레이블
        pbar.update(1) 
        if score < best_score:
            best_score = score
            best_params = params
            best_model = xgb_model
    params_list.append(best_params)
    xgb_models.append(best_model)

model_info = [params_list]

with open('config/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

X_valid['pred'] = 0
for i in range(best_k):
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    X_valid.loc[X_valid_cluster.index, 'pred'] = xgb_models[i].predict(X_valid_cluster.drop(columns=['pred']))

valid_mae = mean_absolute_error(y_valid * valid_df['area_m2'], X_valid['pred'] * X_valid['area_m2'])
print(valid_mae)