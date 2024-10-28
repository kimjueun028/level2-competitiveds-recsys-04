import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('data/train.csv',index_col=0)

df['year_month'] = df['contract_year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int) 
df['month'] = df['year_month'].str[4:].astype(int)  
df['day'] = df['contract_day'].astype(int)
tmp = df['year_month'].unique()
tmp.sort()
mapping = {value:index for index,value in enumerate(tmp)}
df['time_order'] = df['year_month'].map(mapping)
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
start_date = pd.to_datetime('2019-01-01')
df['time_day'] = (df['date'] - start_date).dt.days


when = ['contract_day','contract_year_month']
location = ['latitude','longitude','floor','area_m2','built_year']
df = df.drop_duplicates(subset=location+when+['deposit'])
df = df.drop_duplicates(subset=location+['contract_year_month']+['deposit'])

all_loc = df.drop_duplicates(subset=location).reset_index(drop=True)

i = 0
print('start')
def search_df_fun(x):
    global i
    global df
    global search_df
    global search_loc_df
    
    if i % 100 == 0 :
        print('\n\n\n'+ 'Processing Index : '+ str(i) +' of '+str(all_loc.shape[0]) + '\n\n\n')

    i+=1 
    
    x_loc = x[location]

    search_loc_df = df[
        (df['latitude'] == x_loc['latitude']) &
        (df['longitude'] == x_loc['longitude']) &
        (df['floor'] == x_loc['floor']) &
        (df['area_m2'] == x_loc['area_m2']) &
        (df['built_year'] == x_loc['built_year'])
    ].drop_duplicates(subset=location+['deposit'])
    
    search_loc_df.apply(search_same_df,axis=1)


def search_same_df(x):
    global search_df
    global df
    
    x_loc =x[location+['deposit']]
    
    search_df = df[
        (df['latitude'] == x_loc['latitude']) &
        (df['longitude'] == x_loc['longitude']) &
        (df['floor'] == x_loc['floor']) &
        (df['area_m2'] == x_loc['area_m2']) &
        (df['built_year'] == x_loc['built_year'])&
        (df['deposit'] == x_loc['deposit'])
    ]


    search_df.apply(search_df_del_fun,axis=1)

def search_df_del_fun(x):
    global search_df
    global df

    x_loc =x[location+['deposit']]
    x_time = x['time_day']

    search_df = df[
        (df['latitude'] == x_loc['latitude']) &
        (df['longitude'] == x_loc['longitude']) &
        (df['floor'] == x_loc['floor']) &
        (df['area_m2'] == x_loc['area_m2']) &
        (df['built_year'] == x_loc['built_year'])&
        (df['deposit'] == x_loc['deposit'])
    ]
    
    mask_prev = (search_df['time_day'] <= x_time + 7) & (x_time <= search_df['time_day'])

    
    if np.sum(mask_prev) == 1 :
        return
    
    drop_index = list(search_df[mask_prev].index)
    
    if x.name in drop_index :
        drop_index.remove(x.name)
    
    if drop_index is None :
        return
    else :
        
        df.drop(drop_index,inplace=True)


all_loc.apply(search_df_fun,axis=1)

with open('config/index_list.pkl','wb') as f :
    pickle.dump(list(df.index),f)


