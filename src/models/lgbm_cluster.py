import os
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_df = pd.read_csv("../../data/train_aftercountplace.csv")
test_df = pd.read_csv("../../data/test_aftercountplace.csv")

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

# test 데이터에 대한 cluster 예측
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])

lgb_models = []
best_iterations = []
train_pred = kmeans.predict(X_train[['latitude', 'longitude']])
valid_pred = kmeans.predict(X_valid[['latitude', 'longitude']])
X_train = X_train.drop(columns=['latitude', 'longitude'])
X_valid = X_valid.drop(columns=['latitude', 'longitude'])

lgb_models_params = [
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.582223930507258, 'importance_type': 'split', 'learning_rate': 0.026083382391978157, 'max_depth': 16, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 97, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.4068563410540389, 'reg_lambda': 1.1493273424219395, 'subsample': 0.853801637903136, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.9480313728970867, 'importance_type': 'split', 'learning_rate': 0.09731082211449284, 'max_depth': 19, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 71, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 7.604131423271155, 'reg_lambda': 0.05486459704829388, 'subsample': 0.7604289017236043, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.944623907046247, 'importance_type': 'split', 'learning_rate': 0.05341121408453015, 'max_depth': 19, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 34, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.6915453723709488, 'reg_lambda': 0.7868438053200792, 'subsample': 0.6989836257643808, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.5688190907646915, 'importance_type': 'split', 'learning_rate': 0.038197242016240515, 'max_depth': 17, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 99, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.09604846712192626, 'reg_lambda': 0.011290644716572892, 'subsample': 0.6041609781872814, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.7594733082599556, 'importance_type': 'split', 'learning_rate': 0.029181029572781212, 'max_depth': 10, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 62, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.2411525994245199, 'reg_lambda': 0.3592497957457097, 'subsample': 0.770175759966601, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.8202002311559113, 'importance_type': 'split', 'learning_rate': 0.05188247963537759, 'max_depth': 13, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 92, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.11925219668230191, 'reg_lambda': 0.014489104309926303, 'subsample': 0.6627355127552941, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.8068079507695117, 'importance_type': 'split', 'learning_rate': 0.03047056238307852, 'max_depth': 6, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 60, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.029805675140598887, 'reg_lambda': 0.02596920628459707, 'subsample': 0.711500200540039, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.709165156576725, 'importance_type': 'split', 'learning_rate': 0.03482168550876416, 'max_depth': 13, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 94, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.030584798740038686, 'reg_lambda': 0.04366425202091223, 'subsample': 0.6059766983193776, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.5044049843426449, 'importance_type': 'split', 'learning_rate': 0.04776277339309102, 'max_depth': 17, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 42, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.02526550589849019, 'reg_lambda': 0.25942275542214094, 'subsample': 0.8080646468526838, 'subsample_for_bin': 200000, 'subsample_freq': 0},
    {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.6957680112888753, 'importance_type': 'split', 'learning_rate': 0.02859033206290727, 'max_depth': 12, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 1000, 'n_jobs': -1, 'num_leaves': 84, 'objective': 'regression', 'random_state': 42, 'reg_alpha': 0.22799170337954422, 'reg_lambda': 0.37449600560625945, 'subsample': 0.993362166271533, 'subsample_for_bin': 200000, 'subsample_freq': 0}
]

for i in range(best_k):
    print(f'Cluster {i} modeling...')
    train_cluster_idx = np.where(train_pred == i)[0]   # (index_array, dtype)
    valid_cluster_idx = np.where(valid_pred == i)[0]

    X_train_cluster = X_train.iloc[train_cluster_idx]
    y_train_cluster = y_train.iloc[train_cluster_idx]

    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    y_valid_cluster = y_valid.iloc[valid_cluster_idx]

    lgbm_params = lgb_models_params[i]

    lgb_model = lgb.LGBMRegressor(**lgbm_params)
    lgb_model.fit(X_train_cluster, y_train_cluster, eval_set=[(X_valid_cluster, y_valid_cluster)], eval_metric='l2')
    best_iterations.append(lgb_model.best_iteration_)

    lgb_models.append(lgb_model)

X_valid['pred'] = 0
for i in range(best_k):
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    X_valid.loc[X_valid_cluster.index, 'pred'] = lgb_models[i].predict(X_valid_cluster.drop(columns=['pred']))

valid_pred = X_valid['pred'] * X_valid['area_m2']
valid_mae = mean_absolute_error(y_valid, valid_pred)

print(valid_mae)