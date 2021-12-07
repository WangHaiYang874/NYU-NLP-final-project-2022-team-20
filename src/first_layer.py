# This document assumes that the first layer model is already trained and stored in the first_layer_models
import os
import pickle

class first_layer:
    def __init__(self, model_suffix) -> None:
        
        filepath = '../first_layer_models'
        
        model_names = [i for i in os.listdir(filepath) if model_suffix in i]
        models = dict()
        
        for model_name in model_names:
            with open(filepath + model_names,'rb') as f:
                models[model_name] = pickle.load(f)
                

    def get_probabilities(self, X):
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict_proba(X)
        return results