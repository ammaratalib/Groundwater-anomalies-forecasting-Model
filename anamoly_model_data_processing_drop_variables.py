# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:12:41 2021

@author: Ammara
"""
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from math import*
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble._forest import _generate_unsampled_indices
#from rfpimp import *
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from math import sqrt

from sklearn.model_selection import train_test_split

import warnings
from sklearn.metrics import roc_auc_score
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from pandas import read_csv
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler

from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from sklearn.metrics import r2_score
from keras.layers import Dropout
from keras import optimizers 
from keras.layers.advanced_activations import LeakyReLU 

# define model

model = Sequential()
model.add(LSTM(100,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.25))
model.add(RepeatVector(n_outputs))
model.add(LSTM(50,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=50, verbose=1)  # 

############################################################################################

model = Sequential()
model.add(LSTM(15,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(10,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=50, verbose=1)  # 


##########################################################################################

### anomaly 2 months


os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model")


df_0001=pd.read_csv('df_0001.csv')
df_0801=pd.read_csv('df_0801.csv')
df_1101=pd.read_csv('df_1101.csv')
df_1201=pd.read_csv('df_1201.csv')
df_1601=pd.read_csv('df_1601.csv')
df_1701=pd.read_csv('df_1701.csv')
df_2101=pd.read_csv('df_2101.csv')
df_2201=pd.read_csv('df_2201.csv')
df_2601=pd.read_csv('df_2601.csv')
df_2701=pd.read_csv('df_2701.csv')
df_2801=pd.read_csv('df_2801.csv')
df_2901=pd.read_csv('df_2901.csv')
df_3001=pd.read_csv('df_3001.csv')
df_3401=pd.read_csv('df_3401.csv')
df_04101=pd.read_csv('df_04101.csv')
df_4101=pd.read_csv('df_4101.csv')
df_4201=pd.read_csv('df_4201.csv')
df_4301=pd.read_csv('df_4301.csv')
df_4501=pd.read_csv('df_4501.csv')
df_4701=pd.read_csv('df_4701.csv')
df_4901=pd.read_csv('df_4901.csv')
df_5101=pd.read_csv('df_5101.csv')
df_5301=pd.read_csv('df_5301.csv')
df_5601=pd.read_csv('df_5601.csv')
df_5701=pd.read_csv('df_5701.csv')
df_490001=pd.read_csv('df_490001.csv')



# shift datafor forecasting to previde previous boundary conditions 

df=pd.read_csv('df_0001.csv')
df['ID']='df_0001'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_0001=df


df=pd.read_csv('df_0801.csv')
df['ID']='df_0801'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_0801=df

df=pd.read_csv('df_1101.csv')
df['ID']='df_1101'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_1101=df


df=pd.read_csv('df_1201.csv')
df['ID']='df_1201'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_1201=df


df=pd.read_csv('df_1601.csv')
df['ID']='df_1601'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_1601=df


df=pd.read_csv('df_1701.csv')
df['ID']='df_1701'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_1701=df


df=pd.read_csv('df_2101.csv')
df['ID']='df_2101'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2101=df


df=pd.read_csv('df_2201.csv')
df['ID']='df_2201'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2201=df


df=pd.read_csv('df_2601.csv')
df['ID']='df_2601'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2601=df


df=pd.read_csv('df_2701.csv')
df['ID']='df_2701'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2701=df


df=pd.read_csv('df_2801.csv')
df['ID']='df_2801'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2801=df


df=pd.read_csv('df_2901.csv')
df['ID']='df_2901'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2901=df


df=pd.read_csv('df_3001.csv')
df['ID']='df_3001'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_3001=df


df=pd.read_csv('df_3401.csv')
df['ID']='df_3401'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_3401=df


df=pd.read_csv('df_04101.csv')
df['ID']='df_04101'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_04101=df


df=pd.read_csv('df_4101.csv')
df['ID']='df_4101'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4101=df


df=pd.read_csv('df_4201.csv')
df['ID']='df_4201'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4201=df


df=pd.read_csv('df_4301.csv')
df['ID']='df_4301'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4301=df


df=pd.read_csv('df_4501.csv')
df['ID']='df_4501'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4501=df


df=pd.read_csv('df_4701.csv')
df['ID']='df_4701'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4701=df

df=pd.read_csv('df_4901.csv')
df['ID']='df_4901'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4901=df


df=pd.read_csv('df_5101.csv')
df['ID']='df_5101'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_5101=df

df=pd.read_csv('df_5301.csv')
df['ID']='df_5301'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_5301=df


df=pd.read_csv('df_5601.csv')
df['ID']='df_5601'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_5601=df


df=pd.read_csv('df_5701.csv')
df['ID']='df_5701'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_5701=df


df=pd.read_csv('df_490001.csv')
df['ID']='df_490001'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_490001=df

# merge anamoly data

data=pd.concat((df_0001,df_0801,df_1101,df_1201,df_1601,df_1701,df_2101,df_2201,df_2601,df_2701,df_2801,df_2901,df_3001,df_3401,df_04101,df_4101,df_4201,df_4301,df_4501,df_4701,df_4901,df_5101,df_5301,df_5601,df_5701,df_490001),axis=0)
back=data
#back = back.iloc[: , 1:]
back.index = np.arange(0, len(back))


# save train_test data together in one file

#os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model")   # input data

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation")

#back.to_csv('df_temporal_train_test.csv', index=False, header=True)


# do PCA on train data
###############################################################################################

# bring PCA_covariare static and dynamic data data
# make data ready for LSTM

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation")
df=pd.read_csv('df_temporal_train_test_PCA_cov.csv')
back_stat_dyn=df
df.columns

###################################################################################
## all variables
df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.Prin6_top,df.Prin7_top,
df.Prin8_top,df.Prin9_top,df.Prin10_top,df.Prin11_top,df.Prin12_top,df.Prin13_top,df.Prin14_top,df.prev_anom,
df.for_anom),axis=1)
df.isnull().values.any()
back=df
#back.to_csv('df_temporal_train_test_dynamic_stat.csv', index=False, header=True)

#############################################################################################3

# drop variables 

#  no  meterology 

df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.Prin6_top,df.Prin7_top,
df.Prin8_top,df.Prin9_top,df.Prin10_top,df.Prin11_top,df.Prin12_top,df.Prin13_top,df.Prin14_top,df.prev_anom,
df.for_anom),axis=1)
df.isnull().values.any()
back=df

################################################################################################

#  no  topography

df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.prev_anom,
df.for_anom),axis=1)
df.isnull().values.any()
back=df

#############################################################################################

# no soil

df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.SWE,
df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.Prin6_top,df.Prin7_top,
df.Prin8_top,df.Prin9_top,df.Prin10_top,df.Prin11_top,df.Prin12_top,df.Prin13_top,df.Prin14_top,df.prev_anom,
df.for_anom),axis=1)
df.isnull().values.any()
back=df

###########################################################################################

# no SWE

df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.Prin6_top,df.Prin7_top,
df.Prin8_top,df.Prin9_top,df.Prin10_top,df.Prin11_top,df.Prin12_top,df.Prin13_top,df.Prin14_top,df.prev_anom,
df.for_anom),axis=1)
df.isnull().values.any()
back=df

##########################################################################################
## no boundary condition

df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.Prin6_top,df.Prin7_top,
df.Prin8_top,df.Prin9_top,df.Prin10_top,df.Prin11_top,df.Prin12_top,df.Prin13_top,df.Prin14_top,
df.for_anom),axis=1)
df.isnull().values.any()
back=df

###########################################################################################33

# do  no save data when dropping  variable
#os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation")

#back=pd.read_csv('df_temporal_train_test_dynamic_stat.csv')

mask = (back["calib/valid"]==0)
df = back.loc[mask]
df.index = np.arange(0, len(df))
train=df
train_back=train

df=back
mask = (back["calib/valid"]==1)
df = back.loc[mask]
df.index = np.arange(0, len(df))
test=df
test_back=test


df=train.drop(["TIMESTAMP","Mean_GW_Depth_m", "ID","forecast_date","calib/valid"], axis=1)
values = df.values
train=values

df=test.drop(["TIMESTAMP","Mean_GW_Depth_m", "ID","forecast_date","calib/valid"], axis=1)
values = df.values
test=values
#values = df3.astype('float64')

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


train_y=train_y.reshape(train_y.shape[0],1)
test_y=test_y.reshape(test_y.shape[0],1)
scalerX = StandardScaler().fit(train_X)
scalery = StandardScaler().fit(train_y)
train_X = scalerX.transform(train_X)
train_y = scalery.transform(train_y)
test_X = scalerX.transform(test_X)
test_y = scalery.transform(test_y)

# go ahead
b_X_train=train_X
b_X_test=test_X

b_y_train=train_y
b_y_test=test_y

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

########################################################################################

# start from here
train_X=b_X_train
test_X=b_X_test
train_y=b_y_train
test_y=b_y_test
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_y = train_y.reshape((train_y.shape[0],  train_y.shape[1],1))
test_y = test_y.reshape((test_y.shape[0],  test_y.shape[1],1))

n_features = train_X.shape[2]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

verbose, epochs, batch_size = 1, 100, 50
n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # define model

model = Sequential()
model.add(LSTM(100,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.45))
model.add(RepeatVector(n_outputs))
model.add(LSTM(50,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 


####################################################################################################
yhat = model.predict(test_X)
yhat = yhat.reshape(yhat.shape[0],yhat.shape[1])

inv_yhat = scalery.inverse_transform(yhat)
test_y= test_y.reshape(test_y.shape[0],test_y.shape[1])

inv_y = scalery.inverse_transform(test_y)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
rmse

r2_score(inv_y,inv_yhat)

a=inv_y  # obs
b=inv_yhat  # predic

test_back["obs"]=inv_y
test_back["LSTM_model"]=inv_yhat

plt.plot(inv_y,color='r')
plt.plot(inv_yhat,color='b')

#os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation\results")
#test_back.to_csv('df_temporal_test_results.csv', index=False, header=True)


##########################################################################################

# previous calibrated model

model = Sequential()
model.add(LSTM(100,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.45))
model.add(RepeatVector(n_outputs))
model.add(LSTM(50,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 

# try batch 50
#drop out 0.5
#######################################################################################

# no boundary condition model


##################################################

##pumping landuse model 

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly")


df_2901=pd.read_csv('df_2901.csv')
df_3001=pd.read_csv('df_3001.csv')
df_3401=pd.read_csv('df_3401.csv')
df_4201=pd.read_csv('df_4201.csv')
df_4501=pd.read_csv('df_4501.csv')
df_5601=pd.read_csv('df_5601.csv')
df_490001=pd.read_csv('df_490001.csv')


# shift data for forecasting to previde previous boundary conditions 

df=pd.read_csv('df_2901.csv')
df['ID']='df_2901'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_2901=df
df.isnull().values.any()


df=pd.read_csv('df_3001.csv')
df['ID']='df_3001'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_3001=df
df.isnull().values.any()


df=pd.read_csv('df_3401.csv')
df['ID']='df_3401'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_3401=df
df.isnull().values.any()


df=pd.read_csv('df_4201.csv')
df['ID']='df_4201'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4201=df
df.isnull().values.any()


df=pd.read_csv('df_4501.csv')
df['ID']='df_4501'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_4501=df
df.isnull().values.any()


df=pd.read_csv('df_5601.csv')
df['ID']='df_5601'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_5601=df
df.isnull().values.any()

df=pd.read_csv('df_490001.csv')
df['ID']='df_490001'
df['prev_anom']=df["Mean_GW_Depth_m"]-df["Mean_GW_Depth_m"].mean()
df['for_anom']= df['prev_anom'].shift(-2) ### change it to -4 for 4 month forecast
df["forecast_date"]=df["TIMESTAMP"].shift(-2)
df=df.dropna()
# add validation/calibration id
df["calib/valid"]=1
l = round(.7*len(df.index)) 
df.loc[0:l-1,"calib/valid"] = 0
df_490001=df
df.isnull().values.any()
############################################################################################


# merge anamoly data

data=pd.concat((df_2901,df_3001,df_3401,df_4201,df_4501,df_5601,df_490001),axis=0)

back=data
#back = back.iloc[: , 1:]
back.index = np.arange(0, len(back))


# save train_test data together in one file


os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly\PCA_covar_validation")

back.to_csv('df_temporal_train_test.csv', index=False, header=True)


# do PCA on train data
###############################################################################################

# bring PCA_covariare static and dynamic data data
# make data ready for LSTM

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly\PCA_covar_validation")
df=pd.read_csv('PCA_covariates_validation.csv')
back_stat_dyn=df
df.columns



################################################################################################
## all variables
df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.landuse,df.pump,df.prev_anom,df.for_anom),axis=1)
df.isnull().values.any()
back=df

#################################################################################333
## drop variables both pump landuse



df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.prev_anom,df.for_anom),axis=1)
df.isnull().values.any()
back=df

###################################################################################33
# drop pump

df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.landuse,df.prev_anom,df.for_anom),axis=1)
df.isnull().values.any()
back=df

#################################################################################3

# no landuse
df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.pump,df.prev_anom,df.for_anom),axis=1)
df.isnull().values.any()
back=df

##################################################################################

### no boundary
df=pd.concat((df.TIMESTAMP,df.Mean_GW_Depth_m,df.ID,df.forecast_date,df["calib/valid"],df.Prin1_met,df.Prin2_met,df.Prin3_met,df.Prin4_met,df.Prin5_met,df.Prin6_met,df.Prin7_met,df.RO,df.SOIL,df.SWE,
df.ROCKTYPE1,df.ROCKTYPE2,df.avg_clay,df.avg_sand,df.avg_bd,df.Prin1_top,df.Prin2_top,
df.Prin3_top,df.Prin4_top,df.Prin5_top,df.landuse,df.pump,df.for_anom),axis=1)
df.isnull().values.any()
back=df

#####################################################################################33
#back.to_csv('df_temporal_train_test_dynamic_stat.csv', index=False, header=True)


# do not use define directory line or save data when dropping variables
# start from here for pumping model

#os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly\PCA_covar_validation")

#back=pd.read_csv('df_temporal_train_test_dynamic_stat.csv')

mask = (back["calib/valid"]==0)
df = back.loc[mask]
df.index = np.arange(0, len(df))
train=df
train_back=train

df=back
mask = (back["calib/valid"]==1)
df = back.loc[mask]
df.index = np.arange(0, len(df))
test=df
test_back=test


df=train.drop(["TIMESTAMP","Mean_GW_Depth_m", "ID","forecast_date","calib/valid"], axis=1)
values = df.values
train=values

df=test.drop(["TIMESTAMP","Mean_GW_Depth_m", "ID","forecast_date","calib/valid"], axis=1)
values = df.values
test=values
#values = df3.astype('float64')

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


train_y=train_y.reshape(train_y.shape[0],1)
test_y=test_y.reshape(test_y.shape[0],1)
scalerX = StandardScaler().fit(train_X)
scalery = StandardScaler().fit(train_y)
train_X = scalerX.transform(train_X)
train_y = scalery.transform(train_y)
test_X = scalerX.transform(test_X)
test_y = scalery.transform(test_y)


b_X_train=train_X
b_X_test=test_X

b_y_train=train_y
b_y_test=test_y

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

########################################################################################

# start here

train_X=b_X_train
test_X=b_X_test
train_y=b_y_train
test_y=b_y_test
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_y = train_y.reshape((train_y.shape[0],  train_y.shape[1],1))
test_y = test_y.reshape((test_y.shape[0],  test_y.shape[1],1))

n_features = train_X.shape[2]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

verbose, epochs, batch_size = 1, 100, 50
n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # define model


model = Sequential()
model.add(LSTM(300,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 

####################################################################################################
yhat = model.predict(test_X)
yhat = yhat.reshape(yhat.shape[0],yhat.shape[1])

inv_yhat = scalery.inverse_transform(yhat)
test_y= test_y.reshape(test_y.shape[0],test_y.shape[1])

inv_y = scalery.inverse_transform(test_y)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
rmse

r2_score(inv_y,inv_yhat)




plt.plot(inv_y,color='r')
plt.plot(inv_yhat,color='b')

a=inv_y  # obs
b=inv_yhat  # predic

test_back["obs"]=inv_y
#test_back["LSTM_model"]=inv_yhat
test_back["LSTM_model"]=inv_yhat



#os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation\results")

#os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly\PCA_covar_validation\results")
#test_back.to_csv('df_temporal_test_results.csv', index=False, header=True)

##############################################################################################


###################################################################################3
model = Sequential()
model.add(LSTM(200,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(100,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 

#################################################################################

# this gave me 0.53 R2
model = Sequential()
model.add(LSTM(300,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.5)) 
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 

###############################################################################

model = Sequential()
model.add(LSTM(500,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(500,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 

###############################################################################3
model = Sequential()
model.add(LSTM(800,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(400,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 
########################################################################

model = Sequential()
model.add(LSTM(1000,input_shape=(n_timesteps, n_features)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))
model.add(RepeatVector(n_outputs))
model.add(LSTM(800,return_sequences=True))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(50)))
model.add(LeakyReLU(alpha=0.1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
#model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_data=(test_X, test_y))
model.fit(train_X, train_y, epochs=100, batch_size=80, verbose=1)  # 
