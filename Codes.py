''' import the necessary packages'''
from __future__ import print_function
import pandas as pd
import math
import numpy as np
import pyproj
import rtree
import glob
import os
from datetime import datetime, timedelta, date
import csv
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import random
from time import sleep
import _thread as thread
import seaborn as sns

'''import the datasets'''
df_test = pd.read_csv('core-poi_all.gz',delimiter=',',header = 0, low_memory=False) # Safegraph locations datasets
df_cor = pd.read_csv('part-all.gz',delimiter='\t',header = None, low_memory=False) # user POI dataset
# Since the POI dataset headers are unclear, relabel them to clarify the meanings
df_cor.columns = [ 'timestamp', 'id', 'device','latitude','longitude','precision','time_difference','7','8','9','10','11','12','13','14','15','16']

# Limit the POI dataset coordinates into the Boston area: 42.3601° N, 71.0589° W
df_coor = df_cor.loc[(df_cor['longitude'] >= -72) & (df_cor['longitude'] <= -71)]
df_coor = df_coor.loc[(df_coor['latitude'] >= 42) & (df_coor['latitude'] <= 43)].reset_index(drop=True)
df_coor['unique'] = [i for i in range(len(df_coor))]
# Limit the Safegraph dataset coordinates into the Boston
rest_coor = df_test.loc[df_test['sub_category'].isin(['Full-Service Restaurants','Limited-Service Restaurants'])]
rest_coor = rest_coor.loc[rest_coor['city'].isin(['Boston'])]

'''data cleaning & wrangling'''
if df_coor.duplicated().sum() > 0:
    print("there are duplicates in the data, already removed")
    df_coor.drop_duplicates()
# limit the precision radius
df_coor = df_coor.loc[df_coor['precision'] <=20]

# Add local time columns for better interpretation
df_coor['loctime'] = df_coor['timestamp'] + df_coor['time_difference']
df_coor['loctime_1'] = pd.to_datetime(df_coor['loctime'],unit = 's')
df_coor = df_coor.assign(lochour = df_coor.loctime_1.dt.round('H'))
df_coor['locdate'] = pd.DatetimeIndex(df_coor['loctime_1']).date
df_coor['month'] = pd.DatetimeIndex(df_coor['loctime_1']).month
df_coor['hour'] = pd.DatetimeIndex(df_coor['loctime_1']).hour
df_coor['locdate'] = pd.to_datetime(df_coor['loctime'],unit='s').apply(lambda x:x.date())

# Select only day-time users as there are no common delivery after midnight
df_coor_day = df_coor.loc[(df_coor['hour']>= 7) & (df_coor['hour'] < 24)].reset_index(drop=True)

rest_coor['longitude']=pd.to_numeric(rest_coor['longitude'])
rest_coor['latitude']=pd.to_numeric(rest_coor['latitude'])

'''Defind a function to find out coordinates within 20 meters of resturants locations'''
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 *1000# km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d
ll = []
for index,rows in df_coor_day.iterrows():
        lat1 = rows['latitude']
        lon1 = rows['longitude']
        for i,r in rest_coor.iterrows():
                k = distance((lat1,lon1),(r['latitude'],r['longitude']))
                if (k <= 20):
                        ll.append(rows['unique'])
# Get POI coordinates near resturants
df_coor_rest = df_coor_day[(df_coor_day['unique'].isin(ll))|(df_coor_day['unique'].isin(ll))]
df_coor_rest.latitude = pd.to_numeric(df_coor_rest.latitude)
df_coor_rest.longitude = pd.to_numeric(df_coor_rest.longitude)
df_coor_rest = df_coor_rest.sort_values(['id','loctime'], ascending = True)

'''get only the first and last record of a id's consecutive staying period at a specfic restaurant.
If a person visit restaurants multiple time within a short period, it indicates patterns of picking up delivery'''

min_stay = 60*1 # 1 minutes is the minimum duration of a pickup
max_stay = 5*60*60 # 5 hours is the maximum duration of a meal

def consec(record):
    timstam = record['loctime']
    lat_same_next = record['latitude'].shift(-1) == record['latitude']
    lat_same_prev = record['latitude'].shift(1) == record['latitude']
    longt_same_next = record['longitude'].shift(-1) == record['longitude']
    longt_same_prev = record['longitude'].shift(1) == record['longitude']    
    coor_same_prev = lat_same_prev & longt_same_prev
    coor_same_next = lat_same_next & longt_same_next
    is_next_night = ((timstam - timstam.shift(-1)).abs() >= min_stay) & ((timstam - timstam.shift(-1)).abs() <= max_stay) & coor_same_next
    is_prev_night = ((timstam - timstam.shift(1)).abs() >= min_stay) & ((timstam - timstam.shift(1)).abs() <= max_stay) & coor_same_prev
    next_same_id = record['id'] == record['id'].shift(-1)
    prev_same_id = record['id'] == record['id'].shift(1)
    record = record[(is_next_night & next_same_id) |(prev_same_id & is_prev_night)]
    record = record.groupby(["id", "latitude", "longitude"])
    regroup = pd.concat([record.head(1), record.tail(1)]).drop_duplicates().sort_values(['id','loctime', 'latitude', 'longitude']).reset_index(drop=True)
    return regroup
#select a 1-day threshold period,because people can eat at a resturant everyday
start_date = int(datetime(year=2017, month=5, day=1).timestamp())
end_date = int(datetime(year=2017, month=6, day=30).timestamp())
current_date = start_date
result = []
a =0
while current_date <= end_date:
    Bost_coor_rest = df_coor_rest[(df_coor_rest['loctime'] >= current_date) & (df_coor_rest['loctime'] < (current_date + 86400))]
    rest_stay = consec(Bost_coor_rest)
    if not rest_stay.empty:
        if a == 0:
            rest_stay.to_csv('test_bos_coor.csv', mode='a+',header = True, index = None)
            a = a + 1
        else:
            rest_stay.to_csv('test_bos_coor.csv', mode='a+',header = False, index = None) # export the data into a separate csv for further implementation
    current_date += 86400

rest_ = pd.read_csv('test_bos_coor.csv').drop(['Unnamed: 0'], axis=1) # read the above csv
rest_stay=rest_.sort_values(['id','latitude','longitude','loctime'])
rest_stay

'''look at how many restaurants an id visits within certain hours'''
rest_stay['length'] = (rest_stay['loctime'].diff())/(60*60) # rest_stay contains arriving timestamp and exiting timestamp for each id
rest_stay['length'] = rest_stay['length'].iloc[1::2].replace(np.nan,0) # arriving row & length of stay
rest_stay['length'] = rest_stay['length'].shift(-1)
rest_stay = rest_stay.dropna(subset = ['length']).reset_index()
print("The average length of all boston resturant stays is:", round(rest_stay["length"].mean(),2),"hour, which is", round(rest_stay["length"].mean()*60,2),"minutes.")
rest_stay = rest_stay[['id','latitude','longitude','loctime','loctime_1','locdate','length']]
rest_stay.loctime_1 = pd.to_datetime(rest_stay.loctime_1, format='%Y-%m-%d %H:%M')
dates = pd.to_datetime(rest_stay.loctime_1, format='%Y-%m-%d %H:%M')
rest_stay = rest_stay.sort_values(['id','loctime_1'])

'''Visulize the pattern'''
rest_stay['length_min'] = rest_stay['length'] * 60
sns.displot(rest_stay, x="length_min")

# group visits of each id within 5 hours. 
# logic suggests 5 hours is the approximate time between two meals. 
# Delivery person visits restaurants 9, 10 times within 5 hours 
start_date = int(datetime(year=2017, month=5, day=20).timestamp()) 
end_date = int(datetime(year=2017, month=6, day=10).timestamp())
current_date = start_date
df = pd.DataFrame()
while current_date <= end_date:
    daily_rest_stay = rest_stay[(rest_stay['loctime'] >= current_date) & (rest_stay['loctime'] < (current_date + 86400))]
    rest_stay_ = rest_stay.groupby(['id', pd.Grouper(key="loctime_1", freq="300min",offset="60min")]).size().to_frame('visits').reset_index()
    rest_stay_ = rest_stay_.sort_values(by='visits')
    df = df.append([rest_stay_]).drop_duplicates()
    current_date += 86400

#maximum visits a id pays to different resturants within 5 hours are 9 times.

df1 = df.groupby(by = ['visits']).size().to_frame('id_count').reset_index()

# Visulize the pattern of number of visits to restaurants
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x = df1['visits']
y = df1['id_count']
ax.bar(x,y)
plt.show()

delivery = df.loc[(df['visits'] >= 5)] # classify delivery workers
delivery['locdate'] = pd.to_datetime(df['loctime_1']).dt.date
rest_['locdate'] = pd.to_datetime(rest_['locdate']).dt.date

new_df = pd.merge(rest_, delivery,  how='inner', left_on=['id','locdate'], right_on = ['id','locdate'])
