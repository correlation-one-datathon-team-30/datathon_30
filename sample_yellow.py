import pandas as pd
import os
import pickle as cp
import numpy as np
import time
from parse_locid_to_latlong import location_id_to_latlong
data_path = os.path.join('..','datathon_data')

for i in [2,3]:
    yellow = pd.read_csv(os.path.join(data_path,'yellow_trips_2014Q'+str(i)+'.csv')).sample(frac=0.01)
    yellow.to_csv('yellow_trips_2014Q'+str(i)+'_sampled_001.csv')

for i in [1,2]:
    yellow = pd.read_csv(os.path.join(data_path,'yellow_trips_2015Q'+str(i)+'.csv')).sample(frac=0.01)
    yellow.to_csv('yellow_trips_2015Q'+str(i)+'_sampled_001.csv')