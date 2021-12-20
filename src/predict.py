import os
import pickle
import pandas as pd
import numpy as np


class predict:
    def __init__(self):
        # feature extractor
        modelName = '../models/features2021-12-14.model'
        with open(modelName,'rb') as f:
            self.feature_extractor = pickle.load(f)

        # first layer model
        self.first_layer = dict()
        path = '../models/first_layer/'
        version = '2021-12-13'
        for model in os.listdir(path):
            if version in model:
                modelName = model[:4]
                with open(path+model,'rb') as f:
                    self.first_layer[modelName] = pickle.load(f)
        
        # second layer model
        self.second_layer = dict()
        path = '../models/second_layer/'
        for model in os.listdir(path):
            with open(path+model,'rb') as f:
                self.second_layer[model[0]] = pickle.load(f)

    def first_layer_output(self,series):
        return np.array([
            self.first_layer[name].predict_proba(series)[:,0] 
            for name in list(sorted(self.first_layer.keys()))]).T
    
    def predict_one(self,sentence):
        
        series = pd.Series([sentence])
        X = self.feature_extractor.get_features(series)
        
        X2 = self.first_layer_output(X)
        ret = ''
        if self.second_layer['E'].predict(X):
            ret += 'E'
        else:
            ret += 'I'
        if self.second_layer['N'].predict(X2):
            ret += 'N'
        else:
            ret += 'S'
        if self.second_layer['T'].predict(X2):
            ret += 'T'
        else:
            ret += 'F'
        if self.second_layer['J'].predict(X2):
            ret += 'J'
        else:
            ret += 'P'
        return ret
    
    def predict(self,series):
        return [self.predict_one(i) for i in series.values]
        
