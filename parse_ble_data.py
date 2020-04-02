import pickle
import numpy as np
import pandas as pd

def get_enconters(data,distance_threshold=10):
    return data[:2]


data = pickle.load(open('./data/data.p','rb'))
data_parsed = data.groupby(['user1','user2']).apply(get_enconters)
print(data_parsed.shape)