#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:18:42 2022

@author: qdouzery
"""

##Import packages
from tensorflow import keras

def Classifier_0(shape, n_layers, n_neurons):
    ##Initialiser modèle
    classifier = keras.models.Sequential(name='Classifier_0')
    
    ##Input layer
    classifier.add(keras.layers.Input(shape))
    
    ##Hidden layers
    for i in range(n_layers):
        classifier.add(keras.layers.Dense(n_neurons, kernel_initializer='uniform', activation='relu'))
        
    ##Output layer
    classifier.add(keras.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid', name='Output'))
    
    ##Compile model
    classifier.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    
    return classifier


def Regressor_1(shape, n_layers, n_neurons): 
    ##Initialize model
    regressor = keras.models.Sequential(name='Regressor_1')
   
    ##Input layer
    regressor.add(keras.layers.Input(shape))
    
    ##Hidden layers
    for i in range(n_layers): 
        regressor.add(keras.layers.Dense(n_neurons, activation='relu'))
   
    ##Output layer
    regressor.add(keras.layers.Dense(1, name='Output'))
    
    ##Compile model
    regressor.compile(optimizer = 'adam',
                      loss      = 'mae',
                      metrics = ['accuracy'])
    
    return regressor