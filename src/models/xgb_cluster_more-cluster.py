import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from src.utils.HuberLoss import custom_loss, custom_metric

# random seed 고정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# data load
train_df = pd.read_csv("../data/train_aftercountplace.csv")
test_df = pd.read_csv("../data/test_aftercountplace.csv")

train_df = train_df.drop(columns=['index'])
test_df = test_df.drop(columns=['index'])

# train, valid split
holdout_start = 202307
holdout_end = 202312
valid_df = train_df[(train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end)]
final_train_df = train_df[~((train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end))]

# X, y split
X_train = final_train_df.drop(columns=['deposit','contract_day'])
y_train = final_train_df['deposit']
X_valid = valid_df.drop(columns=['deposit','contract_day'])
y_valid = valid_df['deposit']
X_test = test_df.drop(columns=['contract_day'])

X_total = train_df.drop(columns=['deposit','contract_day'])
y_total = train_df['deposit']

# feature extraction
area_type = [10.322, 66.12, 99.17, 132.23, 165.29, 317.360]
area_labels = ['small', 'medium-small', 'medium', 'medium-large', 'large']
X_train['area_group'] = pd.cut(X_train['area_m2'], bins=area_type, labels=area_labels)
X_valid['area_group'] = pd.cut(X_valid['area_m2'], bins=area_type, labels=area_labels)
X_total['area_group'] = pd.cut(X_total['area_m2'], bins=area_type, labels=area_labels)
X_test['area_group'] = pd.cut(X_test['area_m2'], bins=area_type, labels=area_labels)

# kmeans clustering
best_k = 10
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED)
kmeans.fit(X_total[['latitude', 'longitude']])
total_pred = kmeans.predict(X_total[['latitude', 'longitude']])
test_pred = kmeans.predict(X_test[['latitude', 'longitude']])

# xgboost modeling
model_list = [['model_{}'.format(i) for i in range(len(area_labels))] for _ in range(best_k)]

train_pred = kmeans.predict(X_train[['latitude', 'longitude']])
valid_pred = kmeans.predict(X_valid[['latitude', 'longitude']])

X_train = X_train.drop(columns=['latitude', 'longitude'])
X_valid = X_valid.drop(columns=['latitude', 'longitude'])

for i in range(best_k):
    print(f'Cluster {i}')
    train_cluster_idx = np.where(train_pred == i)[0]   
    valid_cluster_idx = np.where(valid_pred == i)[0]

    X_train_cluster = X_train.iloc[train_cluster_idx]
    y_train_cluster = y_train.iloc[train_cluster_idx]

    X_valid_cluster = X_valid.iloc[valid_cluster_idx]
    y_valid_cluster = y_valid.iloc[valid_cluster_idx]

    for j, label in enumerate(area_labels):
        print(f'Cluster {i}_{j} modeling')
        
        X_train_cluster_2 = X_train_cluster[X_train_cluster['area_group']==label].drop(columns='area_group')
        y_train_cluster_2 = y_train_cluster.loc[X_train_cluster_2.index]

        X_valid_cluster_2 = X_valid_cluster[X_valid_cluster['area_group']==label].drop(columns='area_group')
        y_valid_cluster_2 = y_valid_cluster.loc[X_valid_cluster_2.index]

        xgb_params = {
            'objective': custom_loss,   
            'eval_metric': custom_metric,   
            'seed': RANDOM_SEED,
            'n_estimators': 1000,   
            'learning_rate': 0.02,   
            'max_depth': 11,
            
            'early_stopping_rounds': 30,
            'n_jobs': -1,
            'device': 'gpu'
        }

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_cluster_2, y_train_cluster_2, eval_set=[(X_train_cluster_2, y_train_cluster_2), (X_valid_cluster_2, y_valid_cluster_2)], verbose=100)
        model_list[i][j] = xgb_model

X_valid['pred'] = 0
for i in range(best_k):
    valid_cluster_idx = np.where(valid_pred == i)[0]
    X_valid_cluster = X_valid.iloc[valid_cluster_idx]

    for j, label in enumerate(area_labels):
        X_valid_cluster_2 = X_valid_cluster[X_valid_cluster['area_group']==label].drop(columns='area_group')
        X_valid.loc[X_valid_cluster_2.index, 'pred'] = model_list[i][j].predict(X_valid_cluster_2.drop(columns=['pred']))

valid_pred = X_valid['pred']
valid_mae = mean_absolute_error(y_valid, valid_pred)

print(valid_mae)