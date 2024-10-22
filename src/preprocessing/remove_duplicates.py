import pandas as pd
import numpy as np
import time
from tqdm import tqdm


ct = time.time()


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_data = pd.read_csv('data/train.csv',index_col=0)


# 데이터 전처리

train_data['year_month'] = train_data['contract_year_month'].astype(str)
train_data['year'] = train_data['year_month'].str[:4].astype(int) 
train_data['month'] = train_data['year_month'].str[4:].astype(int)  
train_data['day'] = train_data['contract_day'].astype(int)
tmp = train_data['year_month'].unique()
tmp.sort()
mapping = {value:index for index,value in enumerate(tmp)}
train_data['time_order'] = train_data['year_month'].map(mapping)

train_data['date'] = pd.to_datetime(train_data[['year', 'month', 'day']])
start_date = pd.to_datetime('2019-01-01')
train_data['time_day'] = (train_data['date'] - start_date).dt.days


df = train_data.copy()


print('Start')

tqdm.pandas()

location = ['latitude','longitude','floor','area_m2','built_year']
all_loc = df.drop_duplicates(subset=location).reset_index(drop=True)


# 입금액이 같은 경우 중복 제거해서 추출
def search_df_fun(x):
    global df
    global search_loc_df
    
    x_loc = x[location]

    search_loc_df = df[
        (df['latitude'] == x_loc['latitude']) &
        (df['longitude'] == x_loc['longitude']) &
        (df['floor'] == x_loc['floor']) &
        (df['area_m2'] == x_loc['area_m2']) &
        (df['built_year'] == x_loc['built_year'])
    ].drop_duplicates(subset=location + ['deposit'])
    
    search_loc_df.apply(search_same_df, axis=1)


# 입금액이 같은 모든 경우에 대해 중복 검사 
def search_same_df(x):
    global search_df
    global df
 

    x_loc = x[location + ['deposit']]

    search_df = df[
        (df['latitude'] == x_loc['latitude']) &
        (df['longitude'] == x_loc['longitude']) &
        (df['floor'] == x_loc['floor']) &
        (df['area_m2'] == x_loc['area_m2']) &
        (df['built_year'] == x_loc['built_year']) &
        (df['deposit'] == x_loc['deposit'])
    ]

    while not search_df.empty:
        search_df_del_fun(search_df.iloc[0, :])



def search_df_del_fun(x):
    global search_df
    global df


    x_time = x['time_day']
    
    mask_prev = (search_df['time_day'] <= x_time + 28) & (x_time <= search_df['time_day'])
    
    drop_index = list(search_df[mask_prev].index)
    
    if drop_index is None:
        print('\n\n\n\n\n ERROR \n\n\n\n\n', drop_index)
        return
    
    elif len(drop_index) == 1:
        search_df.drop(drop_index, inplace=True)
        
    else:
        search_df.drop(drop_index, inplace=True)
        if x.name in drop_index:
            drop_index.remove(x.name)
        df.drop(drop_index, inplace=True)


# tqdm을 사용한 progress bar 추가
all_loc.progress_apply(search_df_fun, axis=1)

print('End')

df.to_csv('../drop_duplicate_train_df.csv')

