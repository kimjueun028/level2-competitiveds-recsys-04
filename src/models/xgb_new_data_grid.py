import os
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV,ParameterGrid
from sklearn.neighbors import BallTree
from src.utils.HuberLoss import custom_loss, custom_metric
import xgboost
from sklearn.cluster import KMeans
# import cupy as cp

with open('config/index_list.pkl', 'rb') as f:
    loaded_list = pickle.load(f)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




train_df = pd.read_csv("data/train_aftercountplace.csv")
test_df = pd.read_csv("data/test_aftercountplace.csv")
sample_submission = pd.read_csv('data/sample_submission.csv')



train_df = train_df.loc[loaded_list,:]


train_df.reset_index(inplace=True,drop=True)


test_df.drop('index',axis=1,inplace=True)


train_df['age'] = train_df['age'].clip(lower=0)

age_mapping = {'0-10': 0, '10-20': 10, '20-30': 20, '30+': 30}
train_df['age_group'] = pd.cut(train_df['age'], bins=[0, 10, 20, 30, 300], labels=[0, 10, 20, 30], right=False)
train_df['age_group'] = train_df['age_group'].astype(int)
test_df['age_group'] = pd.cut(test_df['age'], bins=[0, 10, 20, 30, 300], labels=[0, 10, 20, 30], right=False)
test_df['age_group'] = test_df['age_group'].astype(int)

train_df = train_df.drop(columns=['index', 'contract_day', 'age'])
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


xgb_models = []
best_iterations = []

train_pred = kmeans.predict(X_train[['latitude', 'longitude']])
valid_pred = kmeans.predict(X_valid[['latitude', 'longitude']])
X_train = X_train.drop(columns=['latitude', 'longitude'])
X_valid = X_valid.drop(columns=['latitude', 'longitude'])


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
total_fits = len(param_list) * 5 




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

    # X_train_cluster = cp.array(X_train_cluster)
    # y_train_cluster = cp.array(y_train_cluster)
    # X_valid_cluster = cp.array(X_valid_cluster)
    
    with tqdm(total=total_fits) as pbar:
     for params in param_list:
        fold_scores = []  # 각 fold의 점수를 저장할 리스트
                
        for fold in range(5):  # 5-fold 교차 검증
        # 모델 학습
          xgb_model = xgboost.XGBRegressor(**params)

          xgb_model.fit(X_train_cluster, y_train_cluster, eval_set=[(X_valid_cluster, y_valid_cluster)], verbose=False)
        
          # 예측
          y_pred = xgb_model.predict(X_valid_cluster)
          score = mean_absolute_error(y_valid_cluster, y_pred)  # y_valid는 검증 레이블

          
          fold_scores.append(score)

          pbar.update(1)  # 1회 진전

        # 평균 점수 계산
          
        mean_score = sum(fold_scores) / len(fold_scores)
        # 최적의 모델 업데이트
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_model = xgb_model

    params_list.append(best_params)
    xgb_models.append(best_model)



with open('config/xgb_new_data_params_list.pkl', 'wb') as f:
    pickle.dump(params_list, f)

X_valid['pred'] = 0
for i in range(best_k):
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    X_valid.loc[X_valid_cluster.index, 'pred'] = xgb_models[i].predict(X_valid_cluster.drop(columns=['pred']))


valid_mae = mean_absolute_error(y_valid * valid_df['area_m2'], X_valid['pred'] * X_valid['area_m2'])

print(valid_mae)


xgb_models = []
total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
X_total = X_total.drop(columns=['latitude', 'longitude'])


for i in range(best_k):
    print(f'Cluster {i} modeling...')
    total_cluster_idx = np.where(total_pred == i)[0]   
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

test_pred_xgb_cluster = X_test['pred'].astype(float) * X_test['area_m2'].astype(float)


sample_submission['deposit'] = test_pred_xgb_cluster
sample_submission.to_csv('src/results/output.csv', index=False, encoding='utf-8-sig')

