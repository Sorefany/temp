
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as stats
import pylab as plt
import matplotlib
import os
import time
import io
import requests

from sklearn.linear_model import LinearRegression  #GBM algorithm
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn.ensemble import RandomForestRegressor  #Random Forest algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

from sklearn.externals import joblib #For importing and exporting final GBM model
from sklearn.datasets import load_digits #For importing and exporting final GBM model

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def gbm_implement(taxi_data_url, gbm_file_location):
    pd.options.mode.chained_assignment = None  # default='warn'
    #import data
    raw_req=requests.get(taxi_data_url).content
    raw=pd.read_csv(io.StringIO(raw_req.decode('utf-8')))
    #trim blank space in column names
    raw.columns = [col.strip() for col in raw.columns]
    
    raw['lpep_pickup_datetime'] = pd.to_datetime(raw['lpep_pickup_datetime'])
    raw['Lpep_dropoff_datetime'] = pd.to_datetime(raw['Lpep_dropoff_datetime'])
    
    #Coordiate of JFK airport terminal Area from Google Map
    #(40.652421, -73.796020)
    #(40.652421, -73.773261)
    #(40.638176, -73.773261)
    #(40.638176, -73.796020)
    
    #define jfk_in = 1 if either pickup or dropoff coordinates fall into the square area noted above.
    raw['jfk_in'] = ((raw['Pickup_longitude']>-73.796020) & (raw['Pickup_longitude']<-73.773261) & (raw['Pickup_latitude']>40.638176) & (raw['Pickup_latitude']<40.652421)) | ((raw['Dropoff_longitude']>-73.796020) & (raw['Dropoff_longitude']<-73.773261) & (raw['Dropoff_latitude']>40.638176) & (raw['Dropoff_latitude']<40.652421))
    raw['tip_perc']= raw['Tip_amount']/raw['Total_amount']

    datacc = raw[raw['Payment_type'] == 1 & raw['tip_perc'].notnull()]
    
    #create  hour of day and day of week based on pick up time.
    datacc['pick_hourofday'] = datacc['lpep_pickup_datetime'].dt.hour
    datacc['pick_dayofweek'] = datacc['lpep_pickup_datetime'].dt.dayofweek
    
    #calculate trip time length in the unit minutes.
    datacc['trip_time'] =(datacc['Lpep_dropoff_datetime'] - datacc['lpep_pickup_datetime']).dt.total_seconds()/60
    
    #set missing value for Trip_type to 1.
    datacc.loc[datacc['Trip_type'].isnull(), 'Trip_type'] = 1
    
    #define trip speed;
    datacc['trip_speed'] = datacc['Trip_distance']/datacc['trip_time']
    datacc.loc[datacc['trip_speed'].isnull(), 'trip_speed'] = 0
    datacc.loc[datacc['trip_speed']>999, 'trip_speed'] = 999
    
    #assign binning variables for locations
    pickup_longitude_grid = [-999,	-73.9873,	-73.9722,	-73.9608,	-73.9558,	-73.9517,	-73.9453,	-73.9391,	-73.9226,	-73.8904,	999]
    pickup_latitude_grid =[-999,	40.679,	40.689,	40.7009,	40.7149,	40.7303,	40.755,	40.7888,	40.8051,	40.8163,	999]
    
    datacc['Pickup_longitude_cut'] = pd.cut(datacc['Pickup_longitude'], pickup_longitude_grid)
    datacc['Pickup_latitude_cut'] = pd.cut(datacc['Pickup_latitude'], pickup_latitude_grid)
    
    datacc['Pickup_longitude_cutcode'] = datacc['Pickup_longitude_cut'].cat.codes
    datacc['Pickup_latitude_cutcode'] = datacc['Pickup_latitude_cut'].cat.codes
    datacc['Pickup_location_cutcode'] = datacc['Pickup_longitude_cutcode']*10+datacc['Pickup_latitude_cutcode']

    #change Store_and_fwd_flag to 1 and 0.
    datacc['Store_and_fwd_flag_in'] = (datacc['Store_and_fwd_flag']=='Y')
    
    #Create dummies for location
    datacc = pd.concat([datacc, pd.get_dummies(datacc['Pickup_location_cutcode'], prefix='pickup_location', prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True)], axis=1)
    
    #Devide into 10 by 10 Grid based on percentiles
    dropoff_longitude_grid = [-999,	-73.9933,	-73.9827,	-73.972,	-73.9612,	-73.954,	-73.9464,	-73.937,	-73.9169,	-73.8722,	999]
    dropoff_latitude_grid =[-999,	40.6736,	40.688,	40.7046,	40.7206,	40.7399,	40.7565,	40.7708,	40.7903,	40.8121,	999]

    datacc['Dropoff_longitude_cut'] = pd.cut(datacc['Dropoff_longitude'], dropoff_longitude_grid)
    datacc['Dropoff_latitude_cut'] = pd.cut(datacc['Dropoff_latitude'], dropoff_latitude_grid)
    
    datacc['Dropoff_longitude_cutcode'] = datacc['Dropoff_longitude_cut'].cat.codes
    datacc['Dropoff_latitude_cutcode'] = datacc['Dropoff_latitude_cut'].cat.codes
    datacc['Dropoff_location_cutcode'] = datacc['Dropoff_longitude_cutcode']*10+datacc['Dropoff_latitude_cutcode']

    #Create dummies for location
    datacc = pd.concat([datacc, pd.get_dummies(datacc['Dropoff_location_cutcode'], prefix='dropoff_location', prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True)], axis=1)
    
    #import final GBM model
    predictors = ["VendorID",	"RateCodeID",	"Passenger_count",	"Trip_distance",	"Fare_amount",	"Extra",	"MTA_tax",	"Tolls_amount",	"improvement_surcharge",	"Total_amount",	"Trip_type",	"jfk_in",	"pick_hourofday",	"pick_dayofweek",	"trip_time",	"trip_speed",	"Store_and_fwd_flag_in",	"pickup_location_1",	"pickup_location_2",	"pickup_location_3",	"pickup_location_4",	"pickup_location_5",	"pickup_location_6",	"pickup_location_8",	"pickup_location_9",	"pickup_location_10",	"pickup_location_11",	"pickup_location_12",	"pickup_location_13",	"pickup_location_14",	"pickup_location_16",	"pickup_location_19",	"pickup_location_20",	"pickup_location_21",	"pickup_location_22",	"pickup_location_23",	"pickup_location_24",	"pickup_location_25",	"pickup_location_26",	"pickup_location_27",	"pickup_location_28",	"pickup_location_29",	"pickup_location_30",	"pickup_location_31",	"pickup_location_32",	"pickup_location_33",	"pickup_location_34",	"pickup_location_35",	"pickup_location_36",	"pickup_location_37",	"pickup_location_38",	"pickup_location_39",	"pickup_location_40",	"pickup_location_41",	"pickup_location_42",	"pickup_location_43",	"pickup_location_44",	"pickup_location_45",	"pickup_location_46",	"pickup_location_47",	"pickup_location_48",	"pickup_location_49",	"pickup_location_50",	"pickup_location_51",	"pickup_location_52",	"pickup_location_53",	"pickup_location_54",	"pickup_location_55",	"pickup_location_56",	"pickup_location_57",	"pickup_location_58",	"pickup_location_59",	"pickup_location_60",	"pickup_location_61",	"pickup_location_62",	"pickup_location_63",	"pickup_location_64",	"pickup_location_65",	"pickup_location_66",	"pickup_location_67",	"pickup_location_68",	"pickup_location_69",	"pickup_location_70",	"pickup_location_71",	"pickup_location_72",	"pickup_location_73",	"pickup_location_74",	"pickup_location_75",	"pickup_location_76",	"pickup_location_77",	"pickup_location_78",	"pickup_location_79",	"pickup_location_80",	"pickup_location_81",	"pickup_location_82",	"pickup_location_83",	"pickup_location_84",	"pickup_location_85",	"pickup_location_86",	"pickup_location_87",	"pickup_location_88",	"pickup_location_89",	"pickup_location_90",	"pickup_location_91",	"pickup_location_92",	"pickup_location_93",	"pickup_location_94",	"pickup_location_95",	"pickup_location_96",	"pickup_location_97",	"pickup_location_98",	"pickup_location_99",	"dropoff_location_1",	"dropoff_location_2",	"dropoff_location_3",	"dropoff_location_4",	"dropoff_location_5",	"dropoff_location_6",	"dropoff_location_7",	"dropoff_location_8",	"dropoff_location_9",	"dropoff_location_10",	"dropoff_location_11",	"dropoff_location_12",	"dropoff_location_13",	"dropoff_location_14",	"dropoff_location_15",	"dropoff_location_16",	"dropoff_location_17",	"dropoff_location_18",	"dropoff_location_19",	"dropoff_location_20",	"dropoff_location_21",	"dropoff_location_22",	"dropoff_location_23",	"dropoff_location_24",	"dropoff_location_25",	"dropoff_location_26",	"dropoff_location_27",	"dropoff_location_28",	"dropoff_location_29",	"dropoff_location_30",	"dropoff_location_31",	"dropoff_location_32",	"dropoff_location_33",	"dropoff_location_34",	"dropoff_location_35",	"dropoff_location_36",	"dropoff_location_37",	"dropoff_location_38",	"dropoff_location_39",	"dropoff_location_40",	"dropoff_location_41",	"dropoff_location_42",	"dropoff_location_43",	"dropoff_location_44",	"dropoff_location_45",	"dropoff_location_46",	"dropoff_location_47",	"dropoff_location_48",	"dropoff_location_49",	"dropoff_location_50",	"dropoff_location_51",	"dropoff_location_52",	"dropoff_location_53",	"dropoff_location_54",	"dropoff_location_55",	"dropoff_location_56",	"dropoff_location_57",	"dropoff_location_58",	"dropoff_location_59",	"dropoff_location_60",	"dropoff_location_61",	"dropoff_location_62",	"dropoff_location_63",	"dropoff_location_64",	"dropoff_location_65",	"dropoff_location_66",	"dropoff_location_67",	"dropoff_location_68",	"dropoff_location_69",	"dropoff_location_70",	"dropoff_location_71",	"dropoff_location_72",	"dropoff_location_73",	"dropoff_location_74",	"dropoff_location_75",	"dropoff_location_76",	"dropoff_location_77",	"dropoff_location_78",	"dropoff_location_79",	"dropoff_location_80",	"dropoff_location_81",	"dropoff_location_82",	"dropoff_location_83",	"dropoff_location_84",	"dropoff_location_85",	"dropoff_location_86",	"dropoff_location_87",	"dropoff_location_88",	"dropoff_location_89",	"dropoff_location_90",	"dropoff_location_91",	"dropoff_location_92",	"dropoff_location_93",	"dropoff_location_94",	"dropoff_location_95",	"dropoff_location_96",	"dropoff_location_97",	"dropoff_location_98",	"dropoff_location_99"]
    #if a location indicator is not created due to non-existance, create a variable with zero as its value.
    for p in predictors:
        if p not in datacc.columns:
            datacc[p] = 0
    
    gbm_stored = joblib.load(gbm_file_location)
    datacc['tip_perc_predicted'] = gbm_stored.predict(datacc[predictors])
    print "Model performance for only credit card MAE: " + str((abs(datacc[target].values - datacc['tip_perc_predicted'])).mean())
    print "Model performance for only credit card MSE: " + str(((datacc[target].values - datacc['tip_perc_predicted'])**2).mean())
    
    #merge the predicted variable to the complete raw dataset
    raw = raw.merge(datacc.loc[:,['tip_perc_predicted']], how='left', on=None, left_on=None, right_on=None, left_index=True, right_index=True, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False)
    #predict tip_perc_predicted to be zero when the payment type is non-cash
    raw.loc[((raw['Payment_type'] != 1) & (raw['tip_perc'].notnull())), 'tip_perc_predicted'] = 0
    
    print "Model performance with all payment types MAE: " + str((abs(raw[target].values - raw['tip_perc_predicted'])).mean())
    print "Model performance with all payment types MSE: " + str(((raw[target].values - raw['tip_perc_predicted'])**2).mean())
    
    return raw

#test on Sep 2015 data -- get same performance as the original steps
temp_sep15 = gbm_implement("https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv", "A:\Dropbox\Hunt\C1\gbm_taxi_tip_pred.pkl")
#test on Oct 2015 data -- get same performance as the original steps
temp_oct15 = gbm_implement("https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-10.csv", "A:\Dropbox\Hunt\C1\gbm_taxi_tip_pred.pkl")
#test on Oct 2015 data -- get same performance as the original steps
temp_nov15 = gbm_implement("https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-11.csv", "A:\Dropbox\Hunt\C1\gbm_taxi_tip_pred.pkl")

