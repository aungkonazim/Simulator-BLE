import pandas as pd
import numpy as np
import uuid
import random
from copy import deepcopy
from datetime import datetime,timedelta
from joblib import Parallel,delayed
import pytz

def get_user_ids(n_users,n_samples):
    x = [str(uuid.uuid4()) for i in range(n_users)]*int(n_samples//n_users)
    return x


def get_data_user(user_id,o,users,start_date,end_date,lats,longs,p=.6):
    if np.random.uniform(0,1,1)[0]<p:
        return pd.DataFrame([],columns=['user', 'participant_identifier', 'RSSI', 'distance_estimate','latitude', 'longitude', 'localtime', 'timestamp', 'os', 'count','version'])
    df_col = []
    while start_date<end_date:
        if np.random.uniform(0,1,1)[0]<.1:
            start_date+=timedelta(hours=24)
            continue
        lat = np.random.uniform(lats[0],lats[1],1)[0]
        long = np.random.uniform(longs[0],longs[1],1)[0]
        for h in range(int(np.random.uniform(1,24,1)[0])):
            if np.random.uniform(0,1,1)[0]<p:
                continue
            st = start_date + timedelta(hours=h)
            for i,other_user in enumerate(users[o+1:]):
                if np.random.uniform(0,1,1)[0]<p or other_user==user_id:
                    continue
                temp_df_col = []
                start_time = np.random.uniform(st.timestamp(),(st+timedelta(hours=1)).timestamp(),1)[0]
                len_encounter = np.random.randint(2,60)
                ts_array_full = np.arange(start_time,start_time+len_encounter*60,60) + np.random.normal(1,5,len_encounter)
                ts_array = ts_array_full[random.sample(range(len(ts_array_full)),int((np.random.uniform(50,100,1)[0]*len(ts_array_full))/100))]
                st_lat = np.random.uniform(0.002,.0001,1)[0]
                latitude = np.random.normal(lat,st_lat,len(ts_array))
                st_long = np.random.uniform(0.003,.0002,1)[0]
                longitude = np.random.normal(long,st_long,len(ts_array))
                distance_estimate = np.abs(np.random.normal(np.random.uniform(0.01,2,1)[0],1,len(ts_array)))
                signal_strength = np.array(random.sample(list(range(-100,0))*int(len(ts_array)), int(len(ts_array))))
                signal_strength[distance_estimate<1] = np.random.normal(-20,10,len(distance_estimate[distance_estimate<1]))
                df = pd.DataFrame({'user':[user_id]*len(ts_array),'participant_identifier':[other_user]*len(ts_array),
                                   'RSSI':signal_strength,'distance_estimate':distance_estimate,
                                   'latitude':latitude,'longitude':longitude,'time':ts_array})
                if df.shape[0]==0:
                    continue
                df2 = deepcopy(df)
                df2['user'] = df['participant_identifier']
                df2['participant_identifier'] = df['user']
                df2['time'] = list(df['time'].values)
                final_df = pd.concat([df,df2])
                final_df['time'] = final_df['time'].values + np.random.normal(1,5,final_df.shape[0])
                final_df['distance_estimate'] = final_df['distance_estimate'].values + np.abs(np.random.normal(.5,.1,final_df.shape[0]))
                final_df['localtime'] = final_df['time'].apply(lambda a:datetime.fromtimestamp(a))
                final_df['timestamp'] = final_df['time'].apply(lambda a:datetime.fromtimestamp(a)+timedelta(hours=6))
                final_df['os'] = ['android']*final_df.shape[0]
                final_df['count'] = np.random.randint(1,60,final_df.shape[0])
                final_df['version'] = 1
                final_df['RSSI'] = final_df['RSSI'].astype(np.double)
                final_df['count'] = final_df['count'].astype(np.long)
                final_df['version'] = final_df['version'].astype(np.long)
                final_df.drop(['time'], axis=1,inplace=True)
                df_col.append(final_df)
        start_date+=timedelta(hours=24)
    if len(df_col)>0:
        return pd.concat(df_col).dropna()
    else:
        return pd.DataFrame([],columns=['user', 'participant_identifier', 'RSSI', 'distance_estimate','latitude', 'longitude', 'localtime', 'timestamp', 'os', 'count','version'])

def generate_groups_of_users(users,start_date,end_date,lats,longs,p):
    a = Parallel(n_jobs=min(len(users),30),verbose=5)(delayed(get_data_user)(u,i,users,start_date,end_date,lats,longs,p) for i,u in enumerate(users) if i<len(users)-1)
    return pd.concat(a)

def generate_synthetic_data(n_users=100000,no_of_days = 1):
    start_date = datetime(2017,8,16,0,0,0)
    end_date = datetime(2017,8,16+no_of_days,8,0,0)
    leftdown_latitude = 34.995097
    leftdown_longitude = -90.270083
    rightup_latitude = 35.278829
    rightup_longitude = -89.626866
    lats = [leftdown_latitude,rightup_latitude]
    longs = [leftdown_longitude,rightup_longitude]
    df_col = []
    users = get_user_ids(n_users,n_users)
    major_array = {g:str(i) for i,g in enumerate(np.unique(users))}
    minor_array = {g:str(i) for i,g in enumerate(np.unique(users))}
    if n_users>200:
        ff = n_users//200
        count = 0
        for n in range(ff):
            df = generate_groups_of_users(users[n*200:(n+1)*200],start_date,end_date,lats,longs,np.random.uniform(.5,1,1)[0])
            df_col.append(df)
            count+=200
        if count<n_users:
            df = generate_groups_of_users(users[count:],start_date,end_date,lats,longs,np.random.uniform(.5,1,1)[0])
            df_col.append(df)
    else:
        df = generate_groups_of_users(users,start_date,end_date,lats,longs,np.random.uniform(.5,1,1)[0])
        df_col.append(df)
    final_df = pd.concat(df_col)
    final_df['major'] = final_df['participant_identifier'].apply(lambda a:major_array[a])
    final_df['minor'] = final_df['participant_identifier'].apply(lambda a:major_array[a])
    temp_df = final_df[['participant_identifier','major','minor']].drop_duplicates().reset_index(drop=True)
    temp_df['timestamp'] = [start_date]*temp_df.shape[0]
    temp_df['localtime'] = [start_date]*temp_df.shape[0]
    temp_df['version'] = 1
    temp_df.rename(columns={"participant_identifier": "user"},inplace=True)
    final_df.drop(columns=['participant_identifier'],axis=1,inplace=True)
    final_df.rename(columns={"RSSI": "avg_rssi",'distance_estimate':'avg_distance'},inplace=True)
    print(temp_df.columns,final_df.columns)
    return final_df,temp_df

from IPython import display
data,map_data =  generate_synthetic_data(n_users=2000,no_of_days = 1)