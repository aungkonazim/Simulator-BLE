import pandas as pd
import numpy as np
import uuid
import random
from copy import deepcopy
from datetime import datetime,timedelta

def get_user_ids(n_users,n_samples):
    x = [str(uuid.uuid4()) for i in range(n_users)]*int(n_samples//n_users)
    return x
def get_data(d,n_users,n_samples):
    start_date = datetime(2017,8,d)
    end_date = datetime(2017,8,d+1)
    users_left =  random.sample(get_user_ids(n_users,n_samples),n_samples)
    users_right = random.sample(users_left,n_samples)
    signal_strength = random.sample(list(range(100))*int(n_samples), int(n_samples))
    df = pd.DataFrame({'user1':users_left,'user2':users_right,'signal_strength':signal_strength})
    df = df[df.user1!=df.user2]
    ts_col = np.linspace(start_date.timestamp(),end_date.timestamp(),df.shape[0])
    df['time'] = ts_col + np.random.normal(10,30,df.shape[0])
    df2 = deepcopy(df)
    df2['user1'] = df['user2']
    df2['user2'] = df['user1']
    final_df = pd.concat([df,df2])
    final_df['time'] = pd.to_datetime(final_df['time'],unit='s')
    return final_df

d = 11
n_users = 1000
n_samples = int(1e6)
data = get_data(d,n_users,n_samples).sort_values('time').reset_index(drop=True)
print(data.values[:50])