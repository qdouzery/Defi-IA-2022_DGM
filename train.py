#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 19:33:40 2022

@author: qdouzery
"""

##Import packages
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
import h5py
import os
from tensorflow import keras

##Import files.py
import utils
import models
import preprocess

def Regressor(xtrain, ytrain, xtest, ytest,
              n_layers_r, n_neurons_r,
              epochs_r, batch_size_r,
              to_drop, verbose,
              dict_outliers):
    
    ##Copy original df
    xtrain_cp = xtrain.copy()
    ytrain_cp = ytrain.copy()
    xtest_cp = xtest.copy()
    ytest_cp = ytest.copy()
    
    ##Remove outliers
    xtrain_cp, ytrain_cp = utils.Remove_outliers(xtrain_cp, ytrain_cp, dict_outliers)
    
    ##Drop useless variables
    xtrain_cp.drop(['Id', 'number_sta'], axis=1, inplace=True)
    xtest_cp.drop(['Id'], axis=1, inplace=True)
    ytrain_cp.drop(['Id', 'number_sta'], axis=1, inplace=True)
    
    ##Get 'CoSin month'
    xtrain_CoSinMonth = utils.Variable_to_CoSin(pd.DataFrame(xtrain_cp['month']), "month")
    xtest_CoSinMonth = utils.Variable_to_CoSin(pd.DataFrame(xtest_cp['month']), "month")
    
    ##Drop not wanted variable
    xtrain_cp.drop(to_drop, axis=1, inplace=True)
    xtest_cp.drop(to_drop, axis=1, inplace=True)

    ##Normalization
    xtrain_N, xtest_N = utils.Normalization(xtrain_cp, xtest_cp)
    
    ##Add 'CoSin month'
    xtrain_N = pd.concat([xtrain_N, xtrain_CoSinMonth], axis=1)
    xtest_N = pd.concat([xtest_N, xtest_CoSinMonth], axis=1)
    
    ##Train regressor
    n_variables_r = xtrain_N.shape[1]
    regressor = models.Regressor_1(n_variables_r, n_layers_r, n_neurons_r)
    history_r = regressor.fit(xtrain_N, ytrain_cp, batch_size_r, epochs_r, verbose=verbose)

    ##Regressor predictions
    ypred = ytest_cp.copy()
    ypred = ypred.rename(columns = {'Ground_truth' : 'Prediction'})
    ypred_r = regressor.predict(xtest_N)
    ypred.loc[:,'Prediction'] = ypred_r
    
    ##Round predictions to nearest 10th
    ypred['Prediction'] = ypred['Prediction'].round(1)
        
    return ypred, regressor


if __name__=='__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #desactivate some warning messages
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = '', help='path to folder containing files')
    parser.add_argument('--output_folder', type=str, default = '', help='path to folder to output model & predictions')

    args = parser.parse_args()
    data_path = args.data_path
    output_folder = args.output_folder

    ##Get data path
    xtrain_path = data_path + "/Train/Train/X_station_train.csv"
    xtest_path = data_path + "/Test/Test/X_station_test.csv"
    ytrain_path = data_path + "/Train/Train/Y_train.csv"

    bltest_obs_path = data_path + "/Test/Test/Baselines/Baseline_observation_test.csv"
    bltrain_for_path = data_path + "/Train/Train/Baselines/Baseline_forecast_train.csv"
    bltest_for_path = data_path + "/Test/Test/Baselines/Baseline_forecast_test.csv"

    coords_path = data_path + "/Other/Other/stations_coordinates.csv"

    ##Import data
    coords = pd.read_csv(coords_path)

    xtrain_obs = pd.read_csv(xtrain_path,parse_dates=['date'],infer_datetime_format=True)
    xtrain_obs['number_sta'] = xtrain_obs['number_sta'].astype('category')

    ytrain = pd.read_csv(ytrain_path, parse_dates=['date'], infer_datetime_format=True)
    ytrain['number_sta'] = ytrain['number_sta'].astype('category')

    xtest_obs = pd.read_csv(xtest_path,infer_datetime_format=True)

    bltest_obs = pd.read_csv(bltest_obs_path,infer_datetime_format=True)
    bltrain_for = pd.read_csv(bltrain_for_path, infer_datetime_format=True)
    bltest_for = pd.read_csv(bltest_for_path, infer_datetime_format=True)

    print("##### Data : loaded #####")

    ##Preprocessing parameters
    nan = "fill" #How to handle NaNs
    mean = "all" #How to compute mean variables
    smooth_means = ['season', 'month'] #variables to get smooth mean
    means_on = ['precip'] #on which variable to do the smooth mean

    ##Get preprocess data
    xtrain_p, ytrain_p = preprocess.Preprocess_train(xtrain_obs, ytrain, coords, bltrain_for,
                                                   nan, mean, smooth_means, means_on)

    print("##### Preprocessing xtrain/ytrain : done #####")

    xtest_p = preprocess.Preprocess_test(xtest_obs, coords, bltest_obs, bltest_for,
                                       smooth_means, means_on)

    print("##### Preprocessing xtest : done #####")

    ##Model parameters
    n_layers_r = 20
    n_neurons_r = 32

    ##Training parameters
    verbose = 1
    epochs_r = 20
    batch_size_r = 200

    ##Other parameters
    to_drop = ['month', 'season'] #variables to drop
    dict_outliers = {} #no outliers to remove

    ##Training and predictions
    Regressor_predictions, model = Regressor(xtrain_p, ytrain_p, xtest_p, bltest_obs,
                                           n_layers_r, n_neurons_r,
                                           epochs_r, batch_size_r,
                                           to_drop, verbose,
                                           dict_outliers)

    ##Post processing
    Regressor_predictions['Prediction'] = Regressor_predictions['Prediction'] + 1

    ##Export
    output_file_predictions = "/Predictions_regressor-20x32.csv"
    Regressor_predictions.to_csv(output_folder + output_file_predictions, index=False)

    # %%
    # Méthode 1
    # Requiert h5py (si on utilise cette méthode, rajouter h5py dans le requirements.txt)
    model.save(output_folder+'/model.h5')

    # Méthode 2
    # Requiert pickle (si on utilise cette méthode, rajouter pickle dans le requirements.txt)
    # model_path = 'pickle_model'
    # print(model_path)
    # with open(model_path, 'w+b') as file:
    #      print("Path opened")
    #      print(model)
    #      print(file)
    #      pkl.dump(model, file)
    #      print("File dumped")

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
