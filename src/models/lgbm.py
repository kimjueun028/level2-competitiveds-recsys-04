import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
import sys
sys.path.append('.')
from src.utils.age_group import age_grouping

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Data load
train_df = pd.read_csv("data/train_lgbm.csv")
test_df = pd.read_csv("data/test_lgbm.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

train_df = train_df.drop(columns=['index', 'contract_day', 'age','contract_year_month'])
test_df = test_df.drop(columns=['index', 'contract_day', 'age','contract_year_month'])

# X, y 분리
train_df['deposit_per_area'] = train_df['deposit'] / train_df['area_m2']
y_total = train_df['deposit_per_area']

X_total = train_df.drop(columns=['deposit_per_area', 'deposit'])
y_total = train_df['deposit_per_area']

X_test = test_df.copy()

# k = 10으로 KMeans fit
best_k = 10
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED)
kmeans.fit(X_total[['latitude', 'longitude']])

# train, valid set에 KMeans 적용
lgb_models = []
best_iterations = []

# lgbm 하이퍼파라미터 load
with open('config/lgbm_params.yaml','r') as f :
    config = yaml.safe_load(f)
    lgb_models_params = config['lgb_models_params']

# 최종 학습
lgb_models = []
total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
X_total = X_total.drop(columns=['latitude', 'longitude'])
for i in range(best_k):
    print(f'Cluster {i} modeling...')
    total_cluster_idx = np.where(total_pred == i)[0]   # (index_array, dtype)

    X_total_cluster = X_total.iloc[total_cluster_idx]
    y_total_cluster = y_total.iloc[total_cluster_idx]

    lgbm_params = lgb_models_params[i]

    lgb_model = lgb.LGBMRegressor(**lgbm_params)
    lgb_model.fit(X_total_cluster, y_total_cluster, eval_metric='l2')

    lgb_models.append(lgb_model)

# test 데이터에 대한 cluster 예측
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])

# 최종 예측
X_test['pred'] = 0
X_test = X_test.drop(columns=['latitude', 'longitude'])
for i in range(best_k):
    test_cluster_idx = np.where(test_pred == i)[0]
    X_test_cluster = X_test.iloc[test_cluster_idx]
    X_test.loc[X_test_cluster.index, 'pred'] = lgb_models[i].predict(X_test_cluster.drop(columns=['pred']))

test_pred_xgb_cluster = X_test['pred'] * X_test['area_m2']

sample_submission['deposit'] = test_pred_xgb_cluster
sample_submission.to_csv('results/lgbm.csv', index=False, encoding='utf-8-sig')