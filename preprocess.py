#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:19:51 2022

@author: qdouzery
"""

##Import packages
import pandas as pd
import numpy as np
import utils

def Preprocess_train (xtrain, ytrain, coords, bltrain_for, nan, mean, smooth_means, means_on):
    ##Copy original df
    xtrain_p = xtrain.copy()
    ytrain_p = ytrain.copy()
    coords_c = coords.copy()
    
    ##Add stations coordinates
    xtrain_p = xtrain_p.merge(coords_c, how='left', on='number_sta')
    
    ##Add 'month' variable
    xtrain_p['month'] = xtrain_p['date'].dt.month

    ##Handle NaNs
    if (nan == "drop"): #Drop all NaNs
        xtrain_p.dropna(inplace=True)
        ytrain_p.dropna(inplace=True)
    elif (nan == "fill"): #Fill NaNs
        xtrain_p = xtrain_p.fillna(method="backfill")
        ytrain_p = ytrain_p.fillna(method="backfill")
    
    ##Create 'StationDay' variable
    xtrain_p = xtrain_p.assign(StationDay=pd.Series(np.zeros(xtrain_p.shape[0])).values)
    xtrain_p.loc[:,"StationDay"] = xtrain_p.loc[:,"Id"].apply(utils.Get_StationDay)
    
    ##Create 'PeriodDay' variable
    xtrain_p = xtrain_p.assign(PeriodDay=pd.Series(np.zeros(xtrain_p.shape[0])).values)
    xtrain_p.loc[:,"PeriodDay"] = xtrain_p.loc[:,"Id"].apply(utils.Get_PeriodDay)
    
    ##Compute period mean
    AUX = ['dd', 'hu', 'td', 't', 'ff', 'precip']
    for var in AUX:
        xtrain_p = utils.Mean_period(xtrain_p, var)
    
    ##Create 'LastHour' variable
    xtrain_p = xtrain_p.assign(LastHour=pd.Series(np.zeros(xtrain_p.shape[0])).values)
    xtrain_p.loc[:,'LastHour'] = xtrain_p.loc[:,'Id'].apply(utils.Is_LastHour)
    
    ##Get last hour observation for ['dd', 'hu', 'td', 't', 'ff', 'precip]
    xtrain_p['last_dd'] = xtrain_p['LastHour']*xtrain_p['dd']
    xtrain_p['last_hu'] = xtrain_p['LastHour']*xtrain_p['hu']
    xtrain_p['last_td'] = xtrain_p['LastHour']*xtrain_p['td']
    xtrain_p['last_t'] = xtrain_p['LastHour']*xtrain_p['t']
    xtrain_p['last_ff'] = xtrain_p['LastHour']*xtrain_p['ff']
    xtrain_p['last_precip'] = xtrain_p['LastHour']*xtrain_p['precip']
    
    ##Drop useless variables
    xtrain_p.drop(['date', 'Id', 'LastHour', 'number_sta', 'PeriodDay'], axis=1, inplace=True)
    ytrain_p.drop(['date'], axis=1, inplace=True)
    
    ##Mean on 24 hours for ['dd', 'hu', 'td', 't', 'ff']
    if (mean == "all"): #Mean on all the values of a day
        aux_precip = xtrain_p[['StationDay', 'precip', 'last_dd', 'last_hu', 'last_td', 'last_t', 'last_ff', 'last_precip']]
        sum_precip = aux_precip.groupby(['StationDay'], as_index=False).sum()
        xtrain_p = xtrain_p.groupby(['StationDay'], as_index=False).mean()
        xtrain_p.loc[:,['precip', 'last_dd', 'last_hu', 'last_td', 'last_t', 'last_ff', 'last_precip']] = sum_precip.loc[:,['precip', 'last_dd', 'last_hu', 'last_td', 'last_t', 'last_ff', 'last_precip']]
    elif (mean == "just24"): #Mean just if there are 24 hours in a day
        xtrain_p = xtrain_p.groupby("StationDay", as_index=False).agg(pd.Series.sum, min_count = 24)
        xtrain_p[['dd', 'hu', 'td', 't', 'ff', 'month',
                  'lat', 'lon', 'height_sta']] = xtrain_p[['dd', 'hu', 'td', 't', 'ff', 'month',
                                                           'lat', 'lon', 'height_sta']].divide(24)
    
    ##Drop NaNs
    xtrain_p.dropna(inplace=True)
    
    ##Remove impossible rows (where 'precip' > 93.5)
    xtrain_p = xtrain_p.query('precip <= 93.5')
    xtrain_p.reset_index(drop=True, inplace=True)
    
    ##Keep the same rows in xtrain and ytrain
    xtrain_p = xtrain_p.loc[xtrain_p['StationDay'].isin(ytrain_p["Id"])]
    ytrain_p = ytrain_p.loc[ytrain_p['Id'].isin(xtrain_p["StationDay"])]
    
    ##Reset index
    xtrain_p.reset_index(drop=True, inplace=True)
    ytrain_p.reset_index(drop=True, inplace=True)
    
    ##Sort xtrain (based on ytrain 'Id')
    xtrain_p = xtrain_p.set_index('StationDay')
    xtrain_p = xtrain_p.reindex(index=ytrain_p['Id'])
    xtrain_p = xtrain_p.reset_index()
    
    ##Add 'season' variable
    xtrain_p = xtrain_p.assign(season=pd.Series(np.zeros(xtrain_p.shape[0])).values)
    xtrain_p.loc[:,"season"] = xtrain_p.loc[:,'month'].apply(utils.Get_Season)
    
    ##Get smooth mean for wanted variable
    for variable in smooth_means:
        for variable_bis in means_on:
            mean_variable_on = "mean_" + variable + "_" + variable_bis
            xtrain_p[mean_variable_on] = utils.calc_smooth_mean(xtrain_p, by=variable, on=variable_bis, m=300)
        
    ##Add 'forecast' variable (based on the baseline forecast)
    xtrain_p = xtrain_p.merge(bltrain_for, how='left', on='Id')
    xtrain_p = xtrain_p.rename(columns = {'Prediction':'forecast'})
    xtrain_p.drop(['date'], axis=1, inplace=True)
    xtrain_p = xtrain_p.fillna(method="backfill")
        
    return xtrain_p, ytrain_p


def Preprocess_test (xtest, coords, bltest_obs, bltest_for, smooth_means, means_on):
    ##Copy original df
    xtest_p = xtest.copy()
    coords_c = coords.copy()
    
    ##Create 'number_sta' variable
    xtest_p = xtest_p.assign(number_sta=pd.Series(np.zeros(xtest_p.shape[0])).values)
    xtest_p.loc[:,"number_sta"] = xtest_p.loc[:,"Id"].apply(utils.Get_NumberSta)
    
    ##Add stations coordinates
    xtest_p = xtest_p.merge(coords_c, how='left', on='number_sta')
    
    ##Create 'IntId' variable
    xtest_p = xtest_p.assign(IntId=pd.Series(np.zeros(xtest_p.shape[0])).values)
    xtest_p.loc[:,"IntId"] = xtest_p.loc[:,"Id"].apply(utils.Id_to_int)
    
    ##Sort xtest (based on 'IntId')
    xtest_p = xtest_p.sort_values('IntId', ignore_index=True)
    
    ##Fill NaNs
    xtest_p = xtest_p.fillna(method="backfill")
    
    ##Create 'StationDay' variable
    xtest_p = xtest_p.assign(StationDay=pd.Series(np.zeros(xtest_p.shape[0])).values)
    xtest_p.loc[:,"StationDay"] = xtest_p.loc[:,"Id"].apply(utils.Get_StationDay)
    
    ##Create 'PeriodDay' variable
    xtest_p = xtest_p.assign(PeriodDay=pd.Series(np.zeros(xtest_p.shape[0])).values)
    xtest_p.loc[:,"PeriodDay"] = xtest_p.loc[:,"Id"].apply(utils.Get_PeriodDay)
    
    ##Compute period mean
    AUX = ['dd', 'hu', 'td', 't', 'ff', 'precip']
    for var in AUX:
        xtest_p = utils.Mean_period(xtest_p, var)
    
    ##Create 'LastHour' variable
    xtest_p = xtest_p.assign(LastHour=pd.Series(np.zeros(xtest_p.shape[0])).values)
    xtest_p.loc[:,'LastHour'] = xtest_p.loc[:,'Id'].apply(utils.Is_LastHour)
    
    ##Get last hour observation for ['dd', 'hu', 'td', 't', 'ff']
    xtest_p['last_dd'] = xtest_p['LastHour']*xtest_p['dd']
    xtest_p['last_hu'] = xtest_p['LastHour']*xtest_p['hu']
    xtest_p['last_td'] = xtest_p['LastHour']*xtest_p['td']
    xtest_p['last_t'] = xtest_p['LastHour']*xtest_p['t']
    xtest_p['last_ff'] = xtest_p['LastHour']*xtest_p['ff']
    xtest_p['last_precip'] = xtest_p['LastHour']*xtest_p['precip']
    
    ##Drop useless variables
    xtest_p.drop(['Id', 'IntId', 'number_sta', 'LastHour', 'PeriodDay'], axis=1, inplace=True)

    ##Mean on 24 hours for ['dd', 'hu', 'td', 't', 'ff']
    aux_precip = xtest_p[['StationDay', 'precip', 'last_dd', 'last_hu', 'last_td', 'last_t', 'last_ff', 'last_precip']]
    sum_precip = aux_precip.groupby(['StationDay'], as_index=False).sum()
    xtest_p = xtest_p.groupby(['StationDay'], as_index=False).mean()
    xtest_p.loc[:,['precip', 'last_dd', 'last_hu', 'last_td', 'last_t', 'last_ff', 'last_precip']] = sum_precip.loc[:,['precip', 'last_dd', 'last_hu', 'last_td', 'last_t', 'last_ff', 'last_precip']]
    
    ##Drop NaNs
    xtest_p.dropna(inplace=True)
    
    ##Keep the same rows in xtest and ytest
    xtest_p = xtest_p.loc[xtest_p['StationDay'].isin(bltest_obs["Id"])]

    ##Reset index
    xtest_p.reset_index(drop=True, inplace=True)
    
    ##Sort xtest (based on baseline_obs 'Id')
    xtest_p = xtest_p.set_index('StationDay')
    xtest_p = xtest_p.reindex(index=bltest_obs['Id'])
    xtest_p = xtest_p.reset_index()
    
    ##Rearrange columns order (same as in xtrain)
    xtest_p = xtest_p[['Id', 'ff', 't', 'td', 'hu', 'dd', 'precip', 'lat', 'lon', 'height_sta',
                       'month', 'Night_dd', 'Day_dd', 'Evening_dd', 'Night_hu', 'Day_hu',
                       'Evening_hu', 'Night_td', 'Day_td', 'Evening_td', 'Night_t', 'Day_t',
                       'Evening_t', 'Night_ff', 'Day_ff', 'Evening_ff', 'Night_precip',
                       'Day_precip', 'Evening_precip', 'last_dd', 'last_hu', 'last_td',
                       'last_t', 'last_ff', 'last_precip']]
    
    ##Add 'season' variable
    xtest_p = xtest_p.assign(season=pd.Series(np.zeros(xtest_p.shape[0])).values)
    xtest_p.loc[:,"season"] = xtest_p.loc[:,'month'].apply(utils.Get_Season)
    
    ##Get smooth mean for wanted variable
    for variable in smooth_means:
        for variable_bis in means_on:
            mean_variable_on = "mean_" + variable + "_" + variable_bis
            xtest_p[mean_variable_on] = utils.calc_smooth_mean(xtest_p, by=variable, on=variable_bis, m=300)
        
    ##Add 'forecast' variable (based on the baseline forecast)
    xtest_p = xtest_p.merge(bltest_for, how='left', on='Id')
    xtest_p = xtest_p.rename(columns = {'Prediction':'forecast'})
    xtest_p = xtest_p.fillna(method="backfill")
    
    return xtest_p