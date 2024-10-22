import pandas as pd
import numpy as np
import time


ct = time.time()


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



home = '/data/ephemeral/home/project_bsm'



train_data = pd.read_csv(home+'/data/train.csv',index_col=0)



location = ['latitude','longitude','floor','area_m2','built_year']



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

all_loc = df.drop_duplicates(subset=location).reset_index(drop=True)


print('Start')

i = 0
ct = time.time()
count = 1
def search_df_fun(x):
    global i
    global df
    global search_df
    global search_loc_df
    
    if i % 1000 == 0 :
        print('\n\n\n' + str(i) + '\n\n\n')
        print('time :', time.time()-ct)
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
    global count
    

    x_loc =x[location+['deposit']]
    

    search_df = df[
        (df['latitude'] == x_loc['latitude']) &
        (df['longitude'] == x_loc['longitude']) &
        (df['floor'] == x_loc['floor']) &
        (df['area_m2'] == x_loc['area_m2']) &
        (df['built_year'] == x_loc['built_year'])&
        (df['deposit'] == x_loc['deposit'])
    ]

    while not search_df.empty:  # df가 비어있지 않으면 True
        search_df_del_fun(search_df.iloc[0,:])

        if count % 10000 ==0 :
            print('\n\n\n\n', count , '\n\n\n\n')

def search_df_del_fun(x):
    global search_df
    global df
    global count

    x_time = x['time_day']
    
    mask_prev = (search_df['time_day'] <= x_time + 28) & (x_time <= search_df['time_day'])
    
    print('현재 index :', x.name, '몇개 같은지 :', np.sum(mask_prev), end = ' ')
    
    drop_index = list(search_df[mask_prev].index)

    count += len(drop_index)
    
    if drop_index is None :
        print('\n\n\n\n\n 뭔가 잘못됨 \n\n\n\n\n',drop_index)
        return
    
    elif len(drop_index) == 1 :
        search_df.drop(drop_index,inplace=True)
        print('drop은 해당 열만 :' ,drop_index)
    else :
        print('drop 열 들 ',drop_index)
        search_df.drop(drop_index,inplace=True)
        if x.name in drop_index :
          drop_index.remove(x.name)
        df.drop(drop_index,inplace=True)

        

all_loc.apply(search_df_fun,axis=1)

print(time.time()-ct)


# In[ ]:


df.to_csv(home+'/효율개선.csv')

