import pandas as pd
import os
import pickle as cp
import numpy as np
data_path = os.path.join('..','datathon_data')

NTA = pd.read_csv(os.path.join(data_path,'geographic.csv')).transpose()
NTA_new = pd.DataFrame(columns=['LAT','LONG'],dtype=np.float128)


for id,row in NTA.iterrows():
    longitudesum = 0
    longituden = 0
    latitudesum = 0
    latituden = 0
    for i,item in enumerate(row):
        if pd.notnull(item):
            if (i+1) % 2 == 0: #second items - longitude
                longitudesum += item
                longituden += 1
            else:
                latitudesum += item
                latituden += 1
    NTA_new.set_value(id,'LAT',latitudesum/latituden)
    NTA_new.set_value(id,'LONG',longitudesum/longituden)

cp.dump(NTA_new,open('NTA_new','wb'))