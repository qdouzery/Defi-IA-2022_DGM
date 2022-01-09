#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:08:57 2022

@author: qdouzery
"""

##Import packages
import numpy as np


def Get_NumberSta(id):
    return int(id.split('_')[0])


def Get_StationDay(id):
    return id.split('_')[0] + "_" + id.split('_')[1]


def Id_to_int(id):
    return int(id.split('_')[0] + id.split('_')[1] + id.split('_')[2])


def Get_Season(month):
    if (month in [1,2,3]):
        season = "Hiver"
    elif (month in [4,5,6]):
        season = "Printemps"
    elif (month in [7,8,9]):
        season = "Ete"
    else:
        season = "Automne"
    
    return season


def Get_PeriodDay(id):
    hour = int(id.split('_')[2])
    if (hour < 6):
        period = "Night"
    elif (hour > 5 and hour < 18):
        period = "Day"
    elif (hour > 17):
        period = "Evening"
        
    return period


def Mean_period(dfo, variable):
    ##Copy original df
    df = dfo.copy()
    
    ##Create new column names
    night_v = "Night_" + variable
    day_v = "Day_" + variable
    evening_v = "Evening_" + variable
    
    ##Group by 'StationDay' AND 'PeriodDay'
    period_variable = df.groupby(['StationDay','PeriodDay'], as_index=False)[variable].agg('mean')
    
    ##Separate period variables
    night_variable = period_variable[period_variable['PeriodDay']=="Night"]
    night_variable = night_variable.rename(columns={variable:night_v})
    night_variable = night_variable.drop(['PeriodDay'], axis=1)
    day_variable = period_variable[period_variable['PeriodDay']=="Day"]
    day_variable = day_variable.rename(columns={variable:day_v})
    day_variable = day_variable.drop(['PeriodDay'], axis=1)
    evening_variable = period_variable[period_variable['PeriodDay']=="Evening"]
    evening_variable = evening_variable.rename(columns={variable:evening_v})
    evening_variable = evening_variable.drop(['PeriodDay'], axis=1)
    
    ##Merge to principal df
    df = df.merge(night_variable, how='left', on=['StationDay'])
    df = df.merge(day_variable, how='left', on=['StationDay'])
    df = df.merge(evening_variable, how='left', on=['StationDay'])
    
    return df


def Is_LastHour(id):
    ##Get hour of the given Id
    hour = int(id.split('_')[2])
    
    ##Determine if it is the last hour of the day
    if (hour == 23):
        return 1
    else:
        return 0
    
    
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)


def Remove_outliers(xdf, ydf, dct):
    xdf_cp = xdf.copy()
    ydf_cp = ydf.copy()
    
    ##Remove small and large outliers
    for variable, outliers in dct.items():
        xdf_cp = xdf_cp[xdf_cp[variable] >= outliers[0]] #Remove small ones
        xdf_cp = xdf_cp[xdf_cp[variable] <= outliers[1]] #Remove large ones
    
    ##Keep same rows in xtrain and ytrain
    ydf_cp = ydf_cp[ydf_cp.index.isin(xdf_cp.index)]
    
    ##Reset index
    xdf_cp.reset_index(drop=True, inplace=True)
    ydf_cp.reset_index(drop=True, inplace=True)
    
    return xdf_cp, ydf_cp


def Remove_stations(xdf, ydf, sta_todrop):
    xdf_cp = xdf.copy()
    ydf_cp = ydf.copy()
    
    ##Remove all stations observations in sta_todrop
    xdf_cp = xdf_cp.drop(xdf_cp[xdf_cp['number_sta'].isin(sta_todrop)].index)

    ##Keep same rows in xtrain and ytrain
    ydf_cp = ydf_cp[ydf_cp.index.isin(xdf_cp.index)]
    
    ##Reset index
    xdf_cp.reset_index(drop=True, inplace=True)
    ydf_cp.reset_index(drop=True, inplace=True)
    
    return xdf_cp, ydf_cp


def Variable_to_CoSin(df, variable):
    df2 = df.copy()
    
    ##Create features names
    norm_var = "norm_" + variable
    cos_var = "cos_" + variable
    sin_var = "sin_" + variable
    
    ##Normalize values to match with the 0-2π cycle
    df2[norm_var] = (2*np.pi*df2[variable])/df2[variable].max()
    
    ##Create cos and sin features
    df2[cos_var] = np.cos(df2[norm_var])
    df2[sin_var] = np.sin(df2[norm_var])
    
    ##Drop normalized variable
    df2.drop([variable, norm_var], axis=1, inplace=True)
    
    return df2


def Normalization(x_train, x_test):
    #Copy of the original df
    x_train_c = x_train.copy()
    x_test_c = x_test.copy()
    
    #Calcul de la moyenne et de la variance de l'échantillon train
    mean = x_train_c.mean()
    std  = x_train_c.std()
    
    #Normalisation
    x_train_c = (x_train_c - mean) / std
    x_test_c  = (x_test_c  - mean) / std

    return x_train_c, x_test_c

