import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from src.utils.HuberLoss import custom_loss, custom_metric

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_df = pd.read_csv("../data/train_aftercountplace.csv")
test_df = pd.read_csv("../data/test_aftercountplace.csv")

train_df = train_df.drop(columns=['index'])
test_df = test_df.drop(columns=['index'])

holdout_start = 202307
holdout_end = 202312
valid_df = train_df[(train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end)]
final_train_df = train_df[~((train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end))]

X_train = final_train_df.drop(columns=['deposit'])
y_train = final_train_df['deposit']
X_valid = valid_df.drop(columns=['deposit'])
y_valid = valid_df['deposit']
X_test = test_df.copy()

X_total = train_df.drop(columns=['deposit'])
y_total = train_df['deposit']

# train + valid 데이터로 최적의 k 찾기
best_k = 10
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED)
kmeans.fit(X_total[['latitude', 'longitude']])
total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])


xgb_models = []
best_iterations = []
train_pred = kmeans.predict(X_train[['latitude', 'longitude']])
valid_pred = kmeans.predict(X_valid[['latitude', 'longitude']])

for i in range(best_k):
    print(f'Cluster {i} modeling...')
    train_cluster_idx = np.where(train_pred == i)[0]   # (index_array, dtype)
    valid_cluster_idx = np.where(valid_pred == i)[0]

    X_train_cluster = X_train.iloc[train_cluster_idx]
    y_train_cluster = y_train.iloc[train_cluster_idx]

    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    y_valid_cluster = y_valid.iloc[valid_cluster_idx]

    xgb_params = {
        'objective': custom_loss,   # default : reg:squarederror  # loss- train
        'eval_metric': custom_metric,   # default : rmse # valid
        'seed': RANDOM_SEED,
        'n_estimators': 500,
        'learning_rate': 0.02,   # default : 0.3
        'max_depth': 12,
        # 'subsample': 0.9,
        # 'colsample_bytree': 0.9,
        # 'reg_alpha': 10.0,
        # 'reg_lambda': 10.0,
        'early_stopping_rounds':20,
        'n_jobs': -1,
    }

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_cluster, y_train_cluster, eval_set=[(X_valid_cluster, y_valid_cluster)], verbose=20)
    best_iterations.append(xgb_model.best_iteration)

    xgb_models.append(xgb_model)

X_valid['pred'] = 0
for i in range(best_k):
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    X_valid.loc[X_valid_cluster.index, 'pred'] = xgb_models[i].predict(X_valid_cluster.drop(columns=['pred']))

valid_pred = X_valid['pred']
valid_mae = mean_absolute_error(y_valid, valid_pred)

print(valid_mae)