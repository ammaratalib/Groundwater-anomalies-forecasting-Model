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
import matplotlib.lines as mlines




# figure for effect of pumping 

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly\PCA_covar_validation\results")

df_pump_all=pd.read_csv('df_temporal_test_results.csv')

df=df_pump_all

y1=df["obs"]*0.3048
x1=df["no_pump_no_land"]*0.3048


x2=df["LSTM_model"]*0.3048

plt.rcParams['figure.figsize'] = (3.2,3.1)

#ax.scatter(pre,obs, color="orange",s=4)
fig, ax = plt.subplots()
ax.scatter(x1,y1,color='black',marker='o',s=6,label="No Pumping and LU")
#ax.plot(x1,y1,'.', color="blue",markersize=2,label="Sandy Loam")  ##0.65
#ax.scatter(x1,y1,color='black',marker='o',s=6,label="No Pumping and LU")
#ax.scatter(x2,y1,color='blue',marker='s',s=6,label="Pumping and LU")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper center',prop={'size': 6})
ax.set_title("WCS, 2017-2019, n=242",size=7)
ax.set_ylabel('Observed GW Anamoly (m)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted GW Anamoly  (m)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(7)
ax.xaxis.label.set_size(7) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 6,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(-0.6,0.4)
ax.set_xlim(-0.6,0.4)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
#plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")


plt.tight_layout()

####################################################################################
from matplotlib import dates

### pumping

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\pumping_landuse_GW_anamoly\PCA_covar_validation\results")

df_pump_all=pd.read_csv('df_temporal_test_results.csv')

df=df_pump_all



df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
#df.index = df.index.map(str)
x=df["TIMESTAMP"]


x1=df["obs"]*0.3048
y1=df["no_pump_no_land"]*0.3048

plt.rcParams['figure.figsize'] = (5,3)
fig, ax = plt.subplots()

ax.plot(x,x1, color="black",linestyle='-', linewidth=1,label="Observed")
ax.plot(x,y1, color="blue",linestyle='-', linewidth=1,label="Predicted")
#ax.plot(x,y2, color="green",linestyle='none',marker='o',label="LSTM_16",markersize=0.3)
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.set_title("Training n=574, Testing n= 242")
ax.legend(loc='upper right',prop={'size': 8})

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_ylabel('GW Anamoly (m)')  # we already handled the x-label with ax1
#ax.yaxis.label.set_size(5)
#ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
#ax.set_ylim(0,6)  # MI=200, NE,250 #
#MI 200, NE 300, MN 300, IW 250,WI 250
#ax.set_ylim(319,321)
ax.set_xlim([datetime.date(2017,1,1),datetime.date(2019,12,31)])

## 300 for NE
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
#ax.set_xscale('custom')
#ax.set_xticks(xticks)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
ax = plt.gca()
#plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


############################################################################################

# temporal model
import matplotlib.pyplot as plt

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation\results")

df_temp=pd.read_csv('df_temporal_test_results.csv')


df=df_temp

y1=df["obs"]*0.3048
x1=df["LSTM_model"]*0.3048


#x2=df["LSTM_model"]*0.3048
plt.rcParams['figure.figsize'] = (3.2,3.1)
fig, ax = plt.subplots()
#ax.scatter(pre,obs, color="orange",s=4)


ax.scatter(x1,y1,color='green',marker='o',s=6)
#ax.scatter(x2,y1,color='blue',marker='s',s=6,label="Pumping and LU")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper center',prop={'size': 6})
ax.set_title("WCS, 1963-2020, n=2442",size=7)
ax.set_ylabel('Observed GW Anamoly (m)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted GW Anamoly (m)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(7)
ax.xaxis.label.set_size(7) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 6,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(-0.75,0.75)
ax.set_xlim(-0.75,0.75)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")


plt.tight_layout()


##########################################################################################3

# temporal timeseries

# top
s= (df["obs"][(df['ID']=='df_5301')])

x1= (df["TIMESTAMP"][(df['ID']=='df_5301')])


#midd

#s= (df["obs"][(df['ID']=='df_2901')])

#x1= (df["TIMESTAMP"][(df['ID']=='df_2901')])


######################################################33

###timeseries

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation\results")

df_temp=pd.read_csv('df_temporal_test_results.csv')

df=df_temp
y1=(df["obs"][(df['ID']=='df_5301')])*0.3048
#x1=df["LSTM_model"]*0.3048

x1=(df["LSTM_model"][(df['ID']=='df_5301')])*0.3048

#time=(df["TIMESTAMP"][(df['ID']=='df_5301')])

time = pd.to_datetime(df["TIMESTAMP"][(df['ID']=='df_5301')])
#df.index = df.index.map(str)
x=time

plt.rcParams['figure.figsize'] = (5,3)
fig, ax = plt.subplots()


ax.plot(x,x1, color="black",linestyle='-', linewidth=1,label="Observed")
ax.plot(x,y1, color="blue",linestyle='-', linewidth=1,label="Predicted")
#ax.plot(x,y2, color="green",linestyle='none',marker='o',label="LSTM_16",markersize=0.3)
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.set_title("Validation n= 166")
ax.legend(loc='upper right',prop={'size': 8})

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_ylabel('GW Anamoly (m)')  # we already handled the x-label with ax1
#ax.yaxis.label.set_size(5)
#ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
#ax.set_ylim(0,6)  # MI=200, NE,250 #
#MI 200, NE 300, MN 300, IW 250,WI 250
ax.set_ylim(-0.75,0.75)
ax.set_xlim([datetime.date(1990,1,1),datetime.date(2004,12,31)])

## 300 for NE
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
#ax.set_xscale('custom')
#ax.set_xticks(xticks)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")

plt.tight_layout()


#### scatter


#x2=df["LSTM_model"]*0.3048
plt.rcParams['figure.figsize'] = (3.2,3.1)
fig, ax = plt.subplots()
#ax.scatter(pre,obs, color="orange",s=4)


ax.scatter(x1,y1,color='green',marker='o',s=6)
#ax.scatter(x2,y1,color='blue',marker='s',s=6,label="Pumping and LU")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper center',prop={'size': 6})
ax.set_title("Marathon, 1990-2003, n=166",size=7)
ax.set_ylabel('Observed GW Anamoly (m)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted GW Anamoly (m)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(7)
ax.xaxis.label.set_size(7) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 6,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(-0.75,0.75)
ax.set_xlim(-0.75,0.75)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")


plt.tight_layout()

############################################################################################3



#midd

#s= (df["obs"][(df['ID']=='df_2901')])

#x1= (df["TIMESTAMP"][(df['ID']=='df_2901')])


######################################################33

###timeseries

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation\results")

df_temp=pd.read_csv('df_temporal_test_results.csv')

df=df_temp
y1=(df["obs"][(df['ID']=='df_2901')])*0.3048
#x1=df["LSTM_model"]*0.3048

x1=(df["LSTM_model"][(df['ID']=='df_2901')])*0.3048


time = pd.to_datetime(df["TIMESTAMP"][(df['ID']=='df_2901')])
#df.index = df.index.map(str)
x=time




plt.rcParams['figure.figsize'] = (5,3)
fig, ax = plt.subplots()


ax.plot(x,x1, color="black",linestyle='-', linewidth=1,label="Observed")
ax.plot(x,y1, color="blue",linestyle='-', linewidth=1,label="Predicted")
#ax.plot(x,y2, color="green",linestyle='none',marker='o',label="LSTM_16",markersize=0.3)
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
ax.set_title("Validation n= 50")
ax.legend(loc='upper right',prop={'size': 8})

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_ylabel('GW Anamoly (m)')  # we already handled the x-label with ax1
#ax.yaxis.label.set_size(5)
#ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
#ax.set_ylim(0,6)  # MI=200, NE,250 #
#MI 200, NE 300, MN 300, IW 250,WI 250
ax.set_ylim(-0.6,0.2)
ax.set_xlim([datetime.date(2016,5,1),datetime.date(2021,5,31)])
#ax.set_xlim([datetime.date(2015),datetime.date(2016)])

## 300 for NE
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
#ax.set_xscale('custom')
#ax.set_xticks(xticks)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")

plt.tight_layout()


#### scatter


#x2=df["LSTM_model"]*0.3048
plt.rcParams['figure.figsize'] = (3.2,3.1)
fig, ax = plt.subplots()
#ax.scatter(pre,obs, color="orange",s=4)


ax.scatter(x1,y1,color='green',marker='o',s=6)
#ax.scatter(x2,y1,color='blue',marker='s',s=6,label="Pumping and LU")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper center',prop={'size': 6})
ax.set_title("Portage, 2016-2020, n=50",size=7)
ax.set_ylabel('Observed GW Anamoly (m)')  # we already handled the x-label with ax1
ax.set_xlabel('Predicted GW Anamoly (m)')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(7)
ax.xaxis.label.set_size(7) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 6,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 6,direction='in')
ax.set_ylim(-0.6,0.2)
ax.set_xlim(-0.6,0.2)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")


plt.tight_layout()

#########################################################################################################









