# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:58:12 2023

@author: ammar
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







os.chdir(r"C:\ammara_MD\paper3\Data\GW_anamoly_model\PCA_covariates_validation\results")
df=pd.read_csv('df_temporal_train_test.csv')
back=df
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


df=train

obs=df["obs"]*0.3048
pre=df["for_anom"]*0.3048


def stat (obs,pre):
    nse=NSE(pre,obs)
    wil=will(pre,obs)
    pear=pearsonr(obs,pre)
    MAE=mean_absolute_error(obs,pre)
    RMSE=sqrt(mean_squared_error(obs,pre))
    pbias=(np.sum(pre-obs)/np.sum(obs))*100 
    return nse,wil,pear, MAE, RMSE,pbias