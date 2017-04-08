import pandas as pd
import os
import pickle as cp
import numpy as np
import time
from parse_locid_to_latlong import location_id_to_latlong
data_path = os.path.join('..','datathon_data')

uber_2015 = pd.read_csv(os.path.join(data_path,'uber_trips_2015.csv')).sample(frac=0.01)
uber_2015['pickup_latitude'] = np.float128(0)
uber_2015['pickup_longitude'] = np.float128(0)
errorcounter =0
counter =0

for id,row in uber_2015.iterrows():
    if counter % 1000 == 0:
        print(counter/len(uber_2015))
    try:
        lat,long = location_id_to_latlong(row['pickup_location_id'])
        uber_2015.set_value(id,'pickup_latitude',lat)
        uber_2015.set_value(id,'pickup_longitude',long)
    except KeyError:
        uber_2015.drop(id)
        errorcounter += 1
    counter +=1

print(errorcounter,len(uber_2015))
uber_2015.to_csv('uber_2015_with_latlong_sampled_0.01.csv')