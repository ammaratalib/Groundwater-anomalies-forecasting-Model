# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:49:27 2023

@author: ammar
"""

import datetime
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import numpy as np
from math import sqrt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.dates as mdates

##################################################################################

import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
from scipy.stats.stats import pearsonr

# for excel
#df1 = pd.read_excel('box_whisker.xlsx', 'training_by_site')

def NSE(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
#    s,o = filter_nan(s,o)
    return 1 - ((sum((s-o)**2))/(sum((o-np.mean(o))**2)))


def will(s,o):

#    s,o = filter_nan(s,o)
    return  1-(sum((o - s)**2))/(sum((abs(s-np.mean(o)) +abs(o-np.mean(o)))**2))


def stat (obs,pre):
    nse=NSE(pre,obs)
    wil=will(pre,obs)
    pear=pearsonr(obs,pre)
    MAE=mean_absolute_error(obs,pre)
    RMSE=sqrt(mean_squared_error(obs,pre))
    pbias=(np.sum(pre-obs)/np.sum(obs))*100
    return nse,wil,pear, MAE, RMSE,pbias

def depth_df(df):
    if (df['ID'] =='df_0001'):
        return 4.6
    elif (df['ID'] =='df_04101'):
        return 6.3
    elif (df['ID'] =='df_0801'):
        return 1.9
    elif (df['ID'] =='df_1101'):
        return 11.2
    elif (df['ID'] =='df_1201'):
        return 12.2
    elif (df['ID'] =='df_1601'):
        return 10.0
    elif (df['ID'] =='df_1701'):
        return 3.9
    elif (df['ID'] =='df_2101'):
        return 4.4
    elif (df['ID'] =='df_2201'):
        return 4.5   
    elif (df['ID'] =='df_2601'):
        return 6.3   
    elif (df['ID'] =='df_2701'):
        return 3.8    
    elif (df['ID'] =='df_2801'):
        return 2.8     
    elif (df['ID'] =='df_2901'):
        return 4.0  
    elif (df['ID'] =='df_3001'):
        return 3.0
    elif (df['ID'] =='df_3401'):
        return 4.8   
    elif (df['ID'] =='df_4101'):
        return 10.6
    elif (df['ID'] =='df_4201'):
        return 9.4   
    elif (df['ID'] =='df_4301'):
        return 1.9   
    elif (df['ID'] =='df_4501'):
        return 1.5       
    elif (df['ID'] =='df_4701'):
        return 1.2     
    elif (df['ID'] =='df_490001'):
        return 4.4     
    elif (df['ID'] =='df_4901'):
        return 1.3 
    elif (df['ID'] =='df_5101'):
        return 5.1 
    elif (df['ID'] =='df_5301'):
        return 5.7 
    elif (df['ID'] =='df_5601'):
        return 1.5
    elif (df['ID'] =='df_5701'):
        return 5.9

def status_df(df):
    if (df['depth'] <5):
        return 'shallow'
    elif (df['depth'] >=5):
        return 'deep'

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")

df=pd.read_csv('df_temporal_train_results.csv')
df.isnull().values.any()

df['depth'] = df.apply(depth_df, axis = 1)


df['status'] = df.apply(status_df, axis = 1)

back=df

obs=df["obs"] # already in meters. no need to convert
pre=df["LSTM_model"]

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
back=df
# shallow -2, to 1.9 anomaly
# deep anomaly -1.85 to 2.04

mask = (df["status"]=='shallow')
df_s = df.loc[mask]
df_s.index = np.arange(0, len(df_s))


obs=df_s["obs"] # already in meters. no need to convert
pre=df_s["LSTM_model"]
stat(obs,pre)


mask = (df["status"]=='deep')
df_d = df.loc[mask]
df_d.index = np.arange(0, len(df_d))

obs=df_d["obs"] # already in meters. no need to convert
pre=df_d["LSTM_model"]
stat(obs,pre)


########################################################################################################
## scatter plot temporal overall training 

##################################################################################################


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df_s["obs"],df_s["LSTM_model"],'x', color="green",markersize=5,marker='^',label="Shallow (n= 4056)",mfc='none')
ax.plot(x1,0.8078*x1+0.0031,'--', linewidth=2, color='black', label='Fitted line (Shallow)')#overall trend line 

ax.plot(df_d["obs"],df_d["LSTM_model"],'x', color="coral",markersize=5,marker='o',label="Deep (n= 1628)",mfc='none')
ax.plot(x1,0.9022*x1+0.0046,'--', linewidth=2, color='purple',label='Fitted line (Deep)')#overall trend line 

#ax.plot(CS4["lw_net"],CS4["lwnet_idso"],'x', color="cyan",markersize=6,marker='1',label="US-CS4 (2020)",mfc='none')
#ax.grid(which='major', axis='both', linestyle='--', linewidth=1)
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
#ax.plot(x1,0.8449*x1+0.0048,'-', linewidth=2)#overall trend line 

#y= 0.8078x + 0.0031 shallow
# y = 0.9022x + 0.0046   deep
ax.set_title("Temporal Model, Training: Depth to GW level Anomalies",fontsize=8)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-2.5,2.5])
ax.set_xlim([-2.5,2.5])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.91	0.90	0.97	0.95	0.113	0.160	-1.30

ax.text(0.95, 0.35, 'RMSE = 0.16 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.29, 'MAE = 0.11 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.23, 'NSE = 0.90',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.17, '$\mathregular{R^{2}}$ = 0.91',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.11, 'n = 5684',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


####################################################################################################
## temporal overall testing 

########################################################################################################

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")

df=pd.read_csv('df_temporal_test_results.csv')
df.isnull().values.any()

df['depth'] = df.apply(depth_df, axis = 1)
df['status'] = df.apply(status_df, axis = 1)


obs=df["obs"] # already in meters. no need to convert
pre=df["LSTM_model"]

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
back_t=df

mask = (df["status"]=='shallow')
df_s = df.loc[mask]
df_s.index = np.arange(0, len(df_s))

obs=df_s["obs"] # already in meters. no need to convert
pre=df_s["LSTM_model"]
stat(obs,pre)

mask = (df["status"]=='deep')
df_d = df.loc[mask]
df_d.index = np.arange(0, len(df_d))
obs=df_d["obs"] # already in meters. no need to convert
pre=df_d["LSTM_model"]
stat(obs,pre)

stat(obs,pre)

########################################################################################################
## scatter plot temporal overall testing

##################################################################################################


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df_s["obs"],df_s["LSTM_model"],'x', color="green",markersize=5,marker='^',label="Shallow (n= 1741)",mfc='none')
ax.plot(x1,0.8061*x1+0.037,'--', linewidth=2, color='black', label='Fitted line (Shallow)')#overall trend line 

ax.plot(df_d["obs"],df_d["LSTM_model"],'x', color="coral",markersize=5,marker='o',label="Deep (n= 701)",mfc='none')
ax.plot(x1,0.8875*x1+0.018,'--', linewidth=2, color='purple',label='Fitted line (Deep)')#overall trend line 

#ax.plot(CS4["lw_net"],CS4["lwnet_idso"],'x', color="cyan",markersize=6,marker='1',label="US-CS4 (2020)",mfc='none')
#ax.grid(which='major', axis='both', linestyle='--', linewidth=1)
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
#ax.plot(x1,0.8327*x1-0.033,'-', linewidth=2)#ET before corr
 

#y= y = 0.8061x - 0.037 shallow
# y = 0.8875x + 0.018   deep
ax.set_title("Temporal Model, Testing: Depth to GW level Anomalies",fontsize=8)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-2.5,2.5])
ax.set_xlim([-2.5,2.5])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.84	0.84	0.95	0.92	0.16	0.22	40.52


ax.text(0.95, 0.35, 'RMSE = 0.22 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.29, 'MAE = 0.16 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.23, 'NSE = 0.84',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.17, '$\mathregular{R^{2}}$ = 0.84',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.11, 'n = 2442',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###########################################################################################################

## timeseries for validation best, medium worst well

#######################################################################################################3

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")

df=pd.read_csv('df_temporal_test_results.csv')
df.isnull().values.any()

df['depth'] = df.apply(depth_df, axis = 1)
df['status'] = df.apply(status_df, axis = 1)

back=df
df=back

# df_4501 best

mask = (df['ID'] =='df_4501') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)


#R2 	       NSE	will	pear	MAE	     RMSE	pbias	n
#0.92	       0.92	0.98	0.96	0.207	0.288	12.84	224



plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
#plt.plot(df.index,df.obs,'x', color="purple", markersize=8,marker='*',label="Observed Anomaly",mfc='none')
plt.plot(df.index,df.obs, color="purple", linestyle='--',label="Observed Anomalies",mfc='none')

plt.plot(df.index,df.LSTM_model,'x', color="teal",markersize=6,marker='o',label="Forecasted Anomalies",mfc='none')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([-2.5,2.5])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Temporal Model, Testing: Well_4501, ID # 442810089194501, (2002-2020, n=224) ",fontsize=10, color='green')
#ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)


ax.text(0.85, 0.85, 'RMSE = 0.29 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=11)

ax.text(0.85, 0.78, 'MAE = 0.20 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=11)

ax.text(0.85, 0.71, 'NSE = 0.92',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=11)

ax.text(0.85, 0.64, '$\mathregular{R^{2}}$ = 0.92',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=600))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#############################################################################################
df=back

# 4301 end
mask = (df['ID'] =='df_4301') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)


#R2 	       NSE	will	pear	MAE	     RMSE	pbias	n
#0.33	    0.05	0.75	0.57	0.095	0.118	-94.99	82


plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
#plt.plot(df.index,df.obs,'x', color="purple", markersize=8,marker='*',label="Observed Anomaly",mfc='none')
plt.plot(df.index,df.obs, color="purple", linestyle='--',label="Observed Anomalies",mfc='none')

plt.plot(df.index,df.LSTM_model,'x', color="teal",markersize=6,marker='d',label="Forecasted Anomalies",mfc='none')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([-.6,.6])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Temporal Model, Testing: Well_4301, ID # 442313089474301, (1974-1981, n=82) ",fontsize=10, color='red')
#ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)


ax.text(0.85, 0.85, 'RMSE = 0.12 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.text(0.85, 0.78, 'MAE = 0.09 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.text(0.85, 0.71, 'NSE = 0.05',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.text(0.85, 0.64, '$\mathregular{R^{2}}$ = 0.33',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=300))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()



##########################################################################################################
df=back


#df_3401 medium 

mask = (df['ID'] =='df_3401') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)


#R2 	       NSE	will	pear	MAE	     RMSE	pbias	n
#0.66	      0.62	0.90	0.81	0.102	0.151	10.52	226


plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
#plt.plot(df.index,df.obs,'x', color="purple", markersize=8,marker='*',label="Observed Anomaly",mfc='none')
plt.plot(df.index,df.obs, color="purple", linestyle='--',label="Observed Anomalies",mfc='none')

plt.plot(df.index,df.LSTM_model,'x', color="teal",markersize=8,marker='1',label="Forecasted Anomalies",mfc='none')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='lower left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([-1,.7])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Temporal Model, Testing: Well_3401, ID # 435244089293401, (2002-2020, n=226) ",fontsize=10, color='coral')
ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)


ax.text(0.85, 0.85, 'RMSE = 0.15 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='coral', fontsize=11)

ax.text(0.85, 0.78, 'MAE = 0.10 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='coral', fontsize=11)

ax.text(0.85, 0.71, 'NSE = 0.62',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='coral', fontsize=11)

ax.text(0.85, 0.64, '$\mathregular{R^{2}}$ = 0.66',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='coral', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=600))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##############################################################################################


##############################################################################################
## spatial model 

###############################################################################################3


os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df=pd.read_csv('df_spatial_train_results.csv')
df.isnull().values.any()

df['depth'] = df.apply(depth_df, axis = 1)

df['status'] = df.apply(status_df, axis = 1)

back=df

obs=df["obs"] # already in meters. no need to convert
pre=df["LSTM_model"]

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
back=df

mask = (df["status"]=='shallow')
df_s = df.loc[mask]
df_s.index = np.arange(0, len(df_s))


obs=df_s["obs"] # already in meters. no need to convert
pre=df_s["LSTM_model"]
stat(obs,pre)


mask = (df["status"]=='deep')
df_d = df.loc[mask]
df_d.index = np.arange(0, len(df_d))

obs=df_d["obs"] # already in meters. no need to convert
pre=df_d["LSTM_model"]
stat(obs,pre)


########################################################################################################
## scatter plot spatial overall training 

##################################################################################################


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df_s["obs"],df_s["LSTM_model"],'x', color="crimson",markersize=5,marker='s',label="Shallow (n= 3949)",mfc='none')
ax.plot(x1,0.7324*x1+0.0142,'--', linewidth=2, color='magenta', label='Fitted line (Shallow)')#overall trend line 

ax.plot(df_d["obs"],df_d["LSTM_model"],'x', color="cyan",markersize=7,marker='+',label="Deep (n= 2087)",mfc='none')
ax.plot(x1,0.9203*x1+0.0089,'--', linewidth=2, color='darkorange',label='Fitted line (Deep)')#overall trend line 
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
#ax.plot(x1,0.8433*x1+0.0133,'-', linewidth=2)#overall trend line 
ax.set_title("Spatial Model, Training: Depth to GW level Anomalies",fontsize=8)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-2.5,2.5])
ax.set_xlim([-2.5,2.5])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.89	0.89	0.9687	0.9455	0.108	0.157	116.12

ax.text(0.95, 0.35, 'RMSE = 0.16 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.29, 'MAE = 0.11 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.23, 'NSE = 0.89',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.17, '$\mathregular{R^{2}}$ = 0.89',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.11, 'n = 6036',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#####################################################################################################

# spatial validation overall

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df=pd.read_csv('df_spatial_test_results_drivers.csv')
df.isnull().values.any()

df['depth'] = df.apply(depth_df, axis = 1)
df['status'] = df.apply(status_df, axis = 1)

back=df

obs=df["obs"] # already in meters. no need to convert
pre=df["LSTM_model"]

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
back=df

mask = (df["status"]=='shallow')
df_s = df.loc[mask]
df_s.index = np.arange(0, len(df_s))


obs=df_s["obs"] # already in meters. no need to convert
pre=df_s["LSTM_model"]
stat(obs,pre)


mask = (df["status"]=='deep')
df_d = df.loc[mask]
df_d.index = np.arange(0, len(df_d))

obs=df_d["obs"] # already in meters. no need to convert
pre=df_d["LSTM_model"]
stat(obs,pre)


########################################################################################################
## scatter plot spatial overall validation

##################################################################################################


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df_s["obs"],df_s["LSTM_model"],'x', color="crimson",markersize=5,marker='s',label="Shallow (n= 1848)",mfc='none')
ax.plot(x1,0.5731*x1+0.0042,'--', linewidth=2, color='magenta', label='Fitted line (Shallow)')#overall trend line 

ax.plot(df_d["obs"],df_d["LSTM_model"],'x', color="cyan",markersize=7,marker='+',label="Deep (n= 242)",mfc='none')
ax.plot(x1,0.6998*x1+0.0478,'--', linewidth=2, color='darkorange',label='Fitted line (Deep)')#overall trend line 

line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
#ax.plot(x1,0.5775*x1+0.0017,'-', linewidth=2)#overall trend line

ax.set_title("Spatial Model, Testing: Depth to GW level Anomalies",fontsize=8)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-2.5,2.5])
ax.set_xlim([-2.5,2.5])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias
#0.79	0.73	0.90	0.89	0.258	0.331	-76.83


ax.text(0.95, 0.35, 'RMSE = 0.33 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.29, 'MAE = 0.26 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.23, 'NSE = 0.73',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.17, '$\mathregular{R^{2}}$ = 0.79',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.11, 'n = 2090',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#####################################################################################################

# timeseries for best medium model spatial  model
os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")

df=pd.read_csv('df_spatial_test_results_drivers.csv')
df.isnull().values.any()

df['depth'] = df.apply(depth_df, axis = 1)
df['status'] = df.apply(status_df, axis = 1)

back=df
########################################################################################################

df=back

# df_4501 best

mask = (df['ID'] =='df_4501') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)

#R2 	       NSE	will	pear	MAE	     RMSE	pbias	n
# 0.90	       0.79	0.92	0.95	0.34	0.41


plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
plt.plot(df.index,df.obs, color="green", linestyle='--',label="Observed Anomalies",mfc='none')
plt.plot(df.index,df.LSTM_model,'x', color="lightsalmon",markersize=6,marker='*',label="Forecasted Spatial Anomalies",mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([-2.5,3])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Spatial Model, Testing: Well_4501, ID # 442810089194501, (1958-2020, n=748) ",fontsize=10, color='blue')
#ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)


ax.text(0.85, 0.95, 'RMSE = 0.41 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)

ax.text(0.85, 0.88, 'MAE = 0.34 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)

ax.text(0.85, 0.81, 'NSE = 0.79',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)

ax.text(0.85, 0.74, '$\mathregular{R^{2}}$ = 0.90',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=1900))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#############################################################################################
df=back

# 4301 end
mask = (df['ID'] =='df_3001') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)

#R2 	       NSE	will	pear	MAE	     RMSE	pbias	n
# 0.55	      0.54	0.81	0.74	0.22	0.31

plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
plt.plot(df.index,df.obs, color="green", linestyle='--',label="Observed Anomalies",mfc='none')
plt.plot(df.index,df.LSTM_model,'x', color="lightsalmon",markersize=6,marker='>',label="Forecasted Spatial Anomalies",mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([-2,2])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Spatial Model, Testing: Well_3001, ID # 441452089433001, (1995-2020, n=310) ",fontsize=10, color='violet')
#ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)



ax.text(0.85, 0.95, 'RMSE = 0.31 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='violet', fontsize=11)

ax.text(0.85, 0.88, 'MAE = 0.22 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='violet', fontsize=11)

ax.text(0.85, 0.81, 'NSE = 0.54',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='violet', fontsize=11)

ax.text(0.85, 0.74, '$\mathregular{R^{2}}$ = 0.55',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='violet', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=1000))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


##########################################################################################################
df=back
mask = (df['ID'] =='df_2201') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df.set_index(('TIMESTAMP'), inplace=True)


#R2 	       NSE	will	pear	MAE	     RMSE	pbias	n
#0.63	       0.58	0.81	0.79	0.27	0.33



plt.rcParams['figure.figsize'] = (7.5, 3.3)
fig, ax = plt.subplots()
plt.plot(df.index,df.obs, color="green", linestyle='--',label="Observed Anomalies",mfc='none')
plt.plot(df.index,df.LSTM_model,'x', color="lightsalmon",markersize=6,marker='s',label="Forecasted Spatial Anomalies",mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([-2,2])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Spatial Model, Testing: Well_2201, ID # 441236089272201, (1964-1979, n=184) ",fontsize=10, color='maroon')
ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)

ax.text(0.85, 0.95, 'RMSE = 0.33 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.text(0.85, 0.88, 'MAE = 0.27 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.text(0.85, 0.81, 'NSE = 0.58',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.text(0.85, 0.74, '$\mathregular{R^{2}}$ = 0.63',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=800))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


##############################################################################################

### boxplot for all sattes temporal and spatial model

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023\tables_forGIS")
df=pd.read_csv('temporal_test_all_variables.csv')
df.isnull().values.any()
temp=df

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023\tables_forGIS")
df=pd.read_csv('spatial_test_all_variables.csv')
df.isnull().values.any()
spatial=df


df=temp
R2_t= df['R2']
NSE_t = df['NSE']
MAE_t= df['MAE']
RMSE_t= df['RMSE']


df=spatial
R2_s= df['R2']
NSE_s = df['NSE']
MAE_s= df['MAE']
RMSE_s= df['RMSE']




data = [R2_t,R2_s,NSE_t,NSE_s,MAE_t,MAE_s,RMSE_t,RMSE_s] 

my_dict1={'R2':R2_t,'NSE': NSE_t}

my_dict2={'R2':R2_s,'NSE': NSE_s}

my_dict3={'MAE': MAE_t,'RMSE': RMSE_t}

my_dict4={'MAE': MAE_s,'RMSE': RMSE_s}

#ax1.set_xticklabels(['$\mathregular{R^{2}}$','$\mathregular{R^{2}}$','NSE','NSE','Willmott','Willmott',
 #                  'MAE','MAE','RMSE','RMSE','pbias','pbias'])
c1="C0"
c2="C2"
c3="C1"
c4="C3"
plt.rcParams['figure.figsize'] = (8,6)
fig, ax = plt.subplots()
bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[0,5],notch=False, widths=2, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='teal', linewidth=1),boxprops=dict(facecolor='limegreen',edgecolor='teal'))


bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [2,7],notch=False, widths=2, 
 patch_artist=True, boxprops=dict(facecolor='deeppink',edgecolor='crimson'),whiskerprops = dict(linestyle='--'
               ,color='crimson' , linewidth=1))
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

bp3=ax2.boxplot(my_dict3.values(),sym='.',positions = [10,15],notch=False, widths=2, 
                 patch_artist=True,whiskerprops = dict(linestyle='--'
                          ,color='teal' , linewidth=1),boxprops=dict(facecolor='limegreen',edgecolor='teal'))

bp4=ax2.boxplot(my_dict4.values(),sym='.',positions = [12,17],notch=False, widths=2, 
                 patch_artist=True, boxprops=dict(facecolor='deeppink',edgecolor='crimson'),whiskerprops = dict(linestyle='--'
                        ,color='crimson'   , linewidth=1))

plt.setp(bp1['medians'], color='teal')
plt.setp(bp3['medians'], color='teal')
plt.setp(bp2['medians'], color='crimson')
plt.setp(bp4['medians'], color='crimson')

plt.axvline(x=3.5,linewidth=0.5, color='darkorange', linestyle='--')
plt.axvline(x=8.5,linewidth=0.5, color='darkorange', linestyle='--')
plt.axvline(x=13.5,linewidth=0.5, color='darkorange', linestyle='--')

plt.xticks([1, 6, 11,17], ['$\mathregular{R^{2}}$', 'NSE','MAE','RMSE'])

ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Temporal Model (n= 2442)', 'Spatial Model (n= 2090)'], loc='upper right',prop={'size': 10})
ax.set_ylabel('$\mathregular{R^{2}}$/ NSE')  # we already handled the x-label with ax1
ax2.set_ylabel('MAE / RMSE (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 14,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 14,direction='in')
ax2.tick_params(axis = 'y', which = 'major', labelsize = 14,direction='in')
#ax.set_ylim(-0.02,1)
#ax.set_title("Validation Sites, Potatoes, Corn, Pasture, Wheatgrass: (2019-2022), n=582",fontsize=6)
ax.set_title('Testing: Two month Forecast Anomalies for Depth to GW',fontsize=14)
ax2.set_ylim(0.05,0.5)
ax.yaxis.label.set_size(14)
ax.xaxis.label.set_size(14) #there is no label 
ax2.yaxis.label.set_size(14)

ax2.locator_params(axis='y', nbins=8)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

##########################################################################################################3333
#'R2_met_imp', 'NSE_met_imp',
# 'MAE_met_imp', 'RMSE_met_imp', 


#'R2_soil_imp', 'NSE_soil_imp',
# 'MAE_soil_imp', 'RMSE_soil_imp', 


#'R2_SWE_imp', 'NSE_SWE_imp',
# 'MAE_SWE_imp', 'RMSE_SWE_imp', 


#'R2_top_imp', 'NSE_top_imp',
# 'MAE_top_imp', 'RMSE_top_imp', 

#'R2_bou_imp', 'NSE_bou_imp',
# 'MAE_bou_imp', 'RMSE_bou_imp'],
#dtype='object')

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023\tables_forGIS")
df=pd.read_csv('drivers_comparison.csv')

back=df


my_dict1={'SWE': df.MAE_SWE_imp}
my_dict2={'top': df.MAE_top_imp}
my_dict3={ 'met':df.MAE_met_imp}
my_dict4={'soil': df.MAE_soil_imp}
my_dict5={ 'bound': df.MAE_bou_imp}

#my_dict2={ 'may_a':rmse_may_a.iloc[:, 1],'jun_a': rmse_jun_a.iloc[:, 1], 'jul_a': rmse_jul_a.iloc[:, 1], 'aug_a':  rmse_aug_a.iloc[:, 1]}

#my_dict1={ 'may_b':rmse_may_b.iloc[:, 1],'jun_b': rmse_jun_b.iloc[:, 1], 'jul_b': rmse_jul_b.iloc[:, 1],'aug_b': rmse_aug_b.iloc[:, 1]}
#my_dict2={ 'may_a':rmse_may_a.iloc[:, 1],'jun_a': rmse_jun_a.iloc[:, 1], 'jul_a': rmse_jul_a.iloc[:, 1], 'aug_a':  rmse_aug_a.iloc[:, 1]}




plt.rcParams['figure.figsize'] = (3.5,3)
fig, ax = plt.subplots()

bp1=ax.boxplot(my_dict1.values(),sym='bo',positions =[0],notch=False, widths=3, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='blue', linewidth=1),boxprops=dict(facecolor='white',edgecolor='blue'))


bp2=ax.boxplot(my_dict2.values(),sym='ro',positions = [5],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='white',edgecolor='red'),whiskerprops = dict(linestyle='--'
               ,color='red' , linewidth=1))


bp3=ax.boxplot(my_dict3.values(),sym='o',positions = [10],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='white',edgecolor='orange'),whiskerprops = dict(linestyle='--'
               ,color='orange' , linewidth=1))

bp4=ax.boxplot(my_dict4.values(),sym='go',positions = [15],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='white',edgecolor='green'),whiskerprops = dict(linestyle='--'
               ,color='green' , linewidth=1))

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis


bp5=ax2.boxplot(my_dict5.values(),sym='o',positions = [20],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='white',edgecolor='purple'),whiskerprops = dict(linestyle='--'
               ,color='purple' , linewidth=1))
 
plt.setp(bp1['medians'], color='blue')
plt.setp(bp2['medians'], color='red')
plt.setp(bp3['medians'], color='orange')
plt.setp(bp4['medians'], color='green')

plt.setp(bp5['medians'], color='purple')
plt.xticks([0, 5, 11.6,16.5,21], [u'Δ SWE', u'Δ topography',u'Δ meteorology' , u'Δ soil',u'Δ boundary'])
ax.set_ylabel('% change in MAE, without SWE, meteorology,soil')  # we already handled the x-label with ax1
ax2.set_ylabel('% change in MAE, without boundary conditions')  # we already handled the x-label with ax1
ax2.yaxis.label.set_size(7)
ax.yaxis.label.set_size(7)
#ax2.set_ylabel('pbias')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 7,direction='in')
ax.set_ylim(-60,100)
ax2.set_ylim(-10,700)
ax.set_title("Temporal Model: Effect of Drivers on Anomalies Forecast",fontsize=7)
ax.xaxis.label.set_size(10) #there is no label 
ax.locator_params(axis='y', nbins=7)
ax2.locator_params(axis='y', nbins=7)
plt.axvline(x=2.5,linewidth=0.3, color='coral', linestyle='--')
plt.axvline(x=7.5,linewidth=0.3, color='coral', linestyle='--')
plt.axvline(x=12.5,linewidth=0.3, color='coral', linestyle='--')
plt.axvline(x=17.5,linewidth=0.3, color='coral', linestyle='--')
#plt.axvline(x=23,linewidth=0.15, color='teal', linestyle='--')
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


######################################################################################################
## pumping effect ammara MD

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df=pd.read_csv('pumping_df_temporal_test_results.csv')

obs=df["obs"] # already in meters. no need to convert
pre=df["no_pump_no_land"]

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)



#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df["obs"],df["no_pump_no_land"],'x', color="tomato",markersize=7,marker='*',label="Testing (n= 242)",mfc='none')
ax.plot(x1,0.5388*x1+0.0289,'--', linewidth=2, color='red')#overall trend line 

line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model without land use and pumping drivers",fontsize=8, color='red')

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '10'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-1.8,0.8])
ax.set_xlim([-1.8,0.8])
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias

#0.41	0.20	0.75	0.64	0.287	0.360	-39.228


ax.text(0.95, 0.29, 'RMSE = 0.36 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.text(0.95, 0.23, 'MAE = 0.29 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.text(0.95, 0.17, 'NSE = 0.20',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

ax.text(0.95, 0.11, '$\mathregular{R^{2}}$ = 0.41',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


##################################################################################################
obs=df["obs"] # already in meters. no need to convert
pre=df["no_pump"] # only land

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df["obs"],df["no_pump"],'x', color="violet",markersize=7,marker='*',label="Testing (n= 242)",mfc='none')
ax.plot(x1,0.5497*x1+0.0115,'--', linewidth=2, color='darkviolet')#overall trend line 
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model with land use driver",fontsize=8, color='darkviolet')
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '10'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-1.8,0.8])
ax.set_xlim([-1.8,0.8])
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias

#0.45	0.23	0.76	0.67	0.282	0.354



ax.text(0.95, 0.29, 'RMSE = 0.35 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)

ax.text(0.95, 0.23, 'MAE = 0.28 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)

ax.text(0.95, 0.17, 'NSE = 0.23',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)

ax.text(0.95, 0.11, '$\mathregular{R^{2}}$ = 0.45',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################################
obs=df["obs"] # already in meters. no need to convert
pre=df["no_landuse"] # only landuse

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)


#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df["obs"],df["no_landuse"],'x', color="orange",markersize=7,marker='*',label="Testing (n= 242)",mfc='none')
ax.plot(x1,0.5458*x1+0.0173,'--', linewidth=2, color='darkorange')#overall trend line 

line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model with pumping driver",fontsize=8, color='darkorange')

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '10'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
#ax.set_ylim([-2.5,2.5])
#ax.set_xlim([-2.5,2.5])
ax.set_ylim([-1.8,0.8])
ax.set_xlim([-1.8,0.8])
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)
#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias

#0.46	0.26	0.77	0.68	0.279	0.346


ax.text(0.95, 0.29, 'RMSE = 0.34 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)

ax.text(0.95, 0.23, 'MAE = 0.27 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)

ax.text(0.95, 0.17, 'NSE = 0.26',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)

ax.text(0.95, 0.11, '$\mathregular{R^{2}}$ = 0.46',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

############################################################################################################3

obs=df["obs"] # already in meters. no need to convert
pre=df["LSTM_model"] # only landuse

x1=obs
y1=pre
x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)



#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df["obs"],df["LSTM_model"],'x', color="turquoise",markersize=7,marker='*',label="Testing (n= 242)",mfc='none')
ax.plot(x1,0.587*x1+0.0066,'--', linewidth=2, color='darkcyan')#overall trend line 

line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model with both land use and pumping drivers",fontsize=8, color='darkcyan')

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '10'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-1.8,0.8])
ax.set_xlim([-1.8,0.8])
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias

#0.53	0.35	0.80	0.73	0.259	0.323



ax.text(0.95, 0.29, 'RMSE = 0.32 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)

ax.text(0.95, 0.23, 'MAE = 0.26 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)

ax.text(0.95, 0.17, 'NSE = 0.35',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)

ax.text(0.95, 0.11, '$\mathregular{R^{2}}$ = 0.53',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

######################################################################################################


# compare side by side 
obs_n=df["obs"] # already in meters. no need to convert
pre_n=df["no_pump_no_land"]

x1_n=obs_n
y1_n=pre_n
x1_n=pd.DataFrame(x1)
y1_n=pd.DataFrame(y1)


obs_a=df["obs"] # already in meters. no need to convert
pre_a=df["LSTM_model"] # only landuse

x1_a=obs_a
y1_a=pre_a
x1_a=pd.DataFrame(x1)
y1_a=pd.DataFrame(y1)

#plt.rcParams['figure.figsize'] = (7,6)
plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(df["obs"],df["no_pump_no_land"],'x', color="tomato",markersize=7,marker='*',label="Model without land use & pumping",mfc='none')
ax.plot(x1_n,0.5388*x1_n+0.0289,'--', linewidth=2, color='red')#overall trend line 


ax.plot(df["obs"],df["LSTM_model"],'x', color="turquoise",markersize=7,marker='*',label="Model with land use & pumping",mfc='none')
ax.plot(x1_a,0.587*x1_a+0.0066,'--', linewidth=2, color='darkcyan')#overall trend line 


line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model, effect of drivers",fontsize=8, color='black')
ax.grid(which='major', axis='both', linestyle='--', linewidth=1)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '8'})
ax.set_xlabel('Observed Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([-1.8,0.8])
ax.set_xlim([-1.8,0.8])

#R2	   NSE	   will	   pear  	MAE  	RMSE	pbias

#0.53	0.35	0.80	0.73	0.259	0.323


plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()





###########################################################################################################

## bar graph for pumping drivers 


# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np



zvals = [9.2,14.1]  # blue *
lvals = [12.7,30.2]  ## black
mvals = [30.1,77.8]  ## green



plt.rcParams['figure.figsize'] = (4, 5)
fig, ax = plt.subplots()
ax.set_ylabel('Percentage change (%) from baseline model', color='black')
N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

rects1 = ax.bar(ind+width*1, zvals, width, color='violet')
rects2 = ax.bar(ind+width*2, lvals, width, color='orange')
rects3 = ax.bar(ind+width*3, mvals, width, color='turquoise')

ax.set_xticks(ind+0.5)

#ax.set_xticks([4, 8], ['R2', 'May', 'Jun','Jul','Aug','Sep','Oct'])

ax.set_xticklabels( ('$\mathregular{R^{2}}$','NSE'))
ax.legend( (rects1[0], rects2[0],rects3[0]), ('Model with landuse', 'Model with pumping','Model with both landuse & pumping') ,loc='upper left',prop={'size':8.5})
ax.set_title("Temporal Model: Effect of landuse and pumping",fontsize=9, color='black')

plt.axhline(y=0,linewidth=1, color='r', linestyle='--')
ax.yaxis.label.set_size(11)
ax.xaxis.label.set_size(14) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
#plt.yticks(np.arange(-0.6,0.8))
#ax.set_ylim(-1,1)
ax.set_ylim(0,100)

ax.text(0.15, 0.12, '9.2',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)

ax.text(0.30, 0.16, '12.7',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)


ax.text(0.43, 0.33, '30.1',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)
######NSE

ax.text(0.66, 0.17, '14.1',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)

ax.text(0.79, 0.33, '30.2',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)


ax.text(0.93, 0.80, '77.8',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)

ax.locator_params(axis='y', nbins=5)

ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

############################################################################################################

zvals = [-1.5,-1.8]  # blue *
lvals = [-2.7,-3.9]  ## black
mvals = [-9.6,-10.3]  ## green



plt.rcParams['figure.figsize'] = (4, 5)
fig, ax = plt.subplots()
ax.set_ylabel('Percentage change (%) from baseline model', color='black')
N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars
rects1 = ax.bar(ind+width*1, zvals, width, color='violet')
rects2 = ax.bar(ind+width*2, lvals, width, color='orange')
rects3 = ax.bar(ind+width*3, mvals, width, color='turquoise')
ax.set_xticks(ind+0.5)
ax.set_xticklabels( ('MAE','RMSE'))
ax.legend( (rects1[0], rects2[0],rects3[0]), ('Model with landuse', 'Model with pumping','Model with both landuse & pumping') ,loc='lower left',prop={'size':8.5})
ax.set_title("Temporal Model: Effect of landuse and pumping",fontsize=9, color='black')
plt.axhline(y=0,linewidth=1, color='r', linestyle='--')
ax.yaxis.label.set_size(11)
ax.xaxis.label.set_size(14) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
#plt.yticks(np.arange(-0.6,0.8))
#ax.set_ylim(-1,1)
ax.set_ylim(-13,-0)
ax.text(0.15, 0.85, '-1.5',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)
ax.text(0.30, 0.75, '-2.7',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)
ax.text(0.43, 0.22, '-9.6',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)
######NSE

ax.text(0.66, 0.83, '-1.8',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkviolet', fontsize=11)
ax.text(0.79, 0.67, '-3.9',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkorange', fontsize=11)
ax.text(0.98, 0.17, '-10.3',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkcyan', fontsize=11)
ax.locator_params(axis='y', nbins=5)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#####################################################################################################

# method section

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df = pd.read_excel('table_m.xlsx', 'temp_explain_results')
####################################################################################################

## autocorrelation, R2 

###########################################################################################3

#autocorr_2
x=df["autocorr_2"]
y=df["R2"]



plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(x,y,'x', color="green",markersize=9,marker='^',mfc='none')
ax.plot(x,0.8395*x+0.0785,'--', linewidth=1, color='green', label='Fitted line')#overall trend line 

line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model, n= 8126",fontsize=12)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Autocorrelation (Lag=2)')  # we already handled the x-label with ax1
ax.set_ylabel('$\mathregular{R^{2}}$')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([0,1])
ax.text(0.59, 0.85, 'Correlation (r) = 0.75',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=11)

ax.text(0.83, 0.61, '3401',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.73, 0.90, '04101',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.21, 0.63, '5601',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.27, 0.27, '4301',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.86, 0.93, '4501',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

########################################################################################################33

## autocorrelation, MAE 

###########################################################################################3

#autocorr_2
x=df["autocorr_2"]
y=df["MAE"]


plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(x,y,'x', color="blueviolet",markersize=9,marker='o',mfc='none')
ax.plot(x,-0.0908*x+0.2198,'--', linewidth=1, color='blueviolet', label='Fitted line')#overall trend line 
line = mlines.Line2D([1, 0], [0,1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model, n= 8126",fontsize=12)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Autocorrelation (Lag=2)')  # we already handled the x-label with ax1
ax.set_ylabel('MAE ($\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([0,0.3])
#ax.set_xlim([0,1])

ax.text(0.97, 0.92, 'Correlation (r) = 0.11',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='blueviolet', fontsize=11)

ax.text(0.80, 0.21, '2601',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.91, 0.73, '4501',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.26, 0.36, '4301',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.16, 0.32, '0801',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


########################################################################################################33

####################################################################################################

## variance, R2 

###########################################################################################3

#4501, 4301, 3401

os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df = pd.read_excel('table_m.xlsx', 'temp_explain_results')

#autocorr_2
x=df["anomalies_var"]
y=df["R2"]




plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(x,y,'x', color="darkturquoise",markersize=8,marker='s',mfc='none')
ax.plot(x,0.5759*x+0.5863,'--', linewidth=1, color='darkturquoise', label='Fitted line')#overall trend line 
line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model, n= 8126",fontsize=12)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Anomalies variance ($\mathregular{\sigma^{2}}$)')  # we already handled the x-label with ax1
ax.set_ylabel('$\mathregular{R^{2}}$')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([0,1])
#ax.set_xlim([0,1])
ax.text(0.97, 0.22, 'Correlation (r) = 0.38',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkturquoise', fontsize=11)

ax.text(0.99, 0.86, '4501',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.12, 0.27, '4301',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.22, 0.62, '3401',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

###################################################################################################


x=df["anomalies_var"]
y=df["MAE"]




plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(x,y,'x', color="orangered",markersize=9,marker='d',mfc='none')
ax.plot(x,0.0518*x+0.1379,'--', linewidth=1, color='orangered', label='Fitted line')#overall trend line 
line = mlines.Line2D([1, 0], [1,0], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model, n= 8126",fontsize=12)

transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper left',prop={'size': '9'})
ax.set_ylabel('MAE ($\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Anomalies variance ($\mathregular{\sigma^{2}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([0,0.3])
#ax.set_xlim([0,1])

ax.text(0.97, 0.92, 'Correlation (r) = 0.04',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orangered', fontsize=11)

ax.text(0.99, 0.74, '4501',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.24, 0.38, '3401',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.12, 0.36, '4301',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


#############################################################################################################


#n=['0001', '0801', '1101', '1201', '1601', '1701', '2101', '2201', '2601', '2701', '2801', '2901', '3001', 
 #   '3401', '04101', '4101', '4201', '4301', '4501', '4701', '4901', '5101', '5301', '5601', '5701', '490001']

#for i, txt in enumerate(n):
 #   ax.annotate(txt, (x[i], y[i]), fontsize=5)


os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df=pd.read_csv('df_temporal_train_results.csv')


os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation")
df=pd.read_csv('df_temporal_train_test_PCA_cov.csv')
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
df['year'] = df['TIMESTAMP'].dt.year
back=df

df=back
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
mask = (df['TIMESTAMP'] >='2010-01-01') & (df['TIMESTAMP'] <= '2020-12-31 23:45:00')
df = df.loc[mask]
df.index = np.arange(0, len(df))



mask = (df['ID'] =='df_3401') 
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df1=df
df=df1

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df.prev_anom, color="salmon", linestyle='--', mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)
#ax.set_ylim([-100,400])
#ax.set_ylim([-10,400])
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

###################################################################################################
#pumping data set
os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\paper_figures\all_results_2023")
df=pd.read_csv('pumping_df_temporal_test_results.csv')

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df.landuse, color="magenta", linestyle='--', mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)
#ax.set_ylim([-100,400])
#ax.set_ylim([-10,400])
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()









plt.plot(df.TMMX*0.1, color="red", linestyle='--',mfc='none')
plt.plot(df.TMMN*0.1, color="green", linestyle='--',mfc='none')
plt.plot(df.SOIL*.1, color="brown", linestyle='--',markersize=9,marker='o', mfc='none')
plt.plot(df.Mean_GW_Depth_m, color="green", linestyle='--',mfc='none')
plt.plot(df.PR, color="blue", linestyle='--',mfc='none')
plt.plot(df.AET*.1, color="purple", linestyle='--',mfc='none')
plt.plot(df.RO, color="", linestyle='--',markersize=9,marker='*', mfc='none')

#########################################################################################################


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df.index,df.Mean_GW_Depth_m, color="green", linestyle='--',label="Observed Anomalies",mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.locator_params(axis='y', nbins=6)
#ax.set_ylim([-2,2])
ax.set_ylabel('Forecasted Anomalies (meter $\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1

ax.set_title("Spatial Model, Testing: Well_2201, ID # 441236089272201, (1964-1979, n=184) ",fontsize=10, color='maroon')
ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize = 10)

ax.text(0.85, 0.95, 'RMSE = 0.33 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.text(0.85, 0.88, 'MAE = 0.27 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.text(0.85, 0.81, 'NSE = 0.58',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.text(0.85, 0.74, '$\mathregular{R^{2}}$ = 0.63',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='maroon', fontsize=11)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=100))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()



























df=back
df=df[(df['ID']=='df_3401')]


plt.plot(df.Mean_GW_Depth_m)

x=df.TIMESTAMP
y=df.Mean_GW_Depth_m



plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(x,y,'x', color="orangered", linestyle='--', mfc='none')

transform = ax.transAxes
line.set_transform(transform)
ax.set_ylabel('MAE ($\mathregular{month^{-1}}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Anomalies variance ($\mathregular{\sigma^{2}}$)')  # we already handled the x-label with ax1

ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 10,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,direction='in')
ax.set_aspect('auto')
ax.set_ylim([0,0.3])
#ax.set_xlim([0,1])

ax.text(0.97, 0.92, 'Correlation (r) = 0.04',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='orangered', fontsize=11)

ax.text(0.99, 0.74, '4501',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

ax.text(0.24, 0.38, '3401',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=9)

plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()





























n=['0001', '0801', '1101', '1201', '1601', '1701', '2101', '2201', '2601', '2701', '2801', '2901', '3001', 
    '3401', '04101', '4101', '4201', '4301', '4501', '4701', '4901', '5101', '5301', '5601', '5701', '490001']

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]), fontsize=5)














#n=['0001', '0801', '1101', '1201', '1601', '1701', '2101', '2201', '2601', '2701', '2801', '2901', '3001', 
 #   '3401', '04101', '4101', '4201', '4301', '4501', '4701', '4901', '5101', '5301', '5601', '5701', '490001']

#for i, txt in enumerate(n):
 #   ax.annotate(txt, (x[i], y[i]), fontsize=5)




plt.rcParams['figure.figsize'] = (4,4)
fig, ax = plt.subplots()
ax.plot(x,y,'x', color="green",markersize=8,marker='^',mfc='none')
ax.plot(x,0.8395*x+0.0785,'--', linewidth=2, color='green', label='Fitted line')#overall trend line 

line = mlines.Line2D([0, 1], [0, 1], color='grey',linestyle='--',linewidth=1)
ax.set_title("Temporal Model, 26 wells, n=8126", fontsize=12)
n=['0001', '0801', '1101', '1201', '1601', '1701', '2101', '2201', '2601', '2701', '2801', '2901', '3001', 
    '3401', '04101', '4101', '4201', '4301', '4501', '4701', '4901', '5101', '5301', '5601', '5701', '490001']

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]), fontsize=5)
    
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#ax.legend(loc='upper left',prop={'size': '9'})
ax.set_xlabel('Autocorrelation (Lag=2)')  # we already handled the x-label with ax1
ax.set_ylabel('$\mathregular{R^{2}}$')  # we already handled the x-label with ax1

ax.text(0.65, 0.95, 'Correlation (r) = 0.75',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=13)
            only_move={'points':'y', 'text':'y'}, force_points=0.15,
            arrowprops=dict(arrowstyle="->", color='r', lw=0.5)
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(12)
ax.xaxis.label.set_size(12) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 12,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 12,direction='in')
ax.set_aspect('auto')
#ax.set_ylim([-2.5,2.5])
ax.set_ylim([0,1])
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()























#y = [0.90, 0.45, 3.52623, 3.51468, 3.02199]
#z = [0.15, 0.3, 0.45, 0.6, 0.75]
#n = [58, 651, 393, 203, 123]

n=[0001, 0801, 1101, 1201, 1601, 1701, 2101, 2201, 2601, 2701, 2801, 2901, 3001, 3401, 04101, 4101, 
   4201, 4301, 4501, 4701, 4901, 5101, 5301, 5601, 5701, 490001]

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))









plt.annotate(label, # this is the tex
             (x,y), # these are the coordinates to position the label
             textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()












ax.text(0.95, 0.35, 'RMSE = 0.16 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.29, 'MAE = 0.11 (m)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.23, 'NSE = 0.90',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.17, '$\mathregular{R^{2}}$ = 0.91',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)

ax.text(0.95, 0.11, 'n = 5684',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=11)




















