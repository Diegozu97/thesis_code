from __future__ import print_function

import platform
import time
import tqdm 
from tqdm import tqdm
tqdm.pandas(desc='My bar!')

# database access
import pandas_datareader as web
import quandl as quandl
import wrds as wrds

# storage and operations
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import joblib

# Visualization Libraries 
import matplotlib.pyplot as plt 
import seaborn as sns

# Statistical Analysis Libraries 
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone # How to create our own scaler 
import statsmodels.api as sm
import linearmodels as lm 
from itertools import product, combinations
# import torch
import xgboost as xgb

# Others 
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count


import warnings 
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

class Backtester:
    
    def __init__(self,
                 df,
                 modeling_features,
                 rolling_frw,
                 look_back_prm,
                 configurations,
                 col_to_pred
                 ):
        
        self.df = df
        self.modeling_features = modeling_features
        self.col_to_pred = col_to_pred
        self.rolling_frq = rolling_frw
        self.look_back_prm = look_back_prm
        self.configurations = configurations
        self.dict_all_predictions = {}
        self.dict_feature_importance = {}
        self.dict_feature_importance["random_forest"] = {}
        self.dict_feature_importance["xgboost"] = {}
        self.model = sm


        for key in configurations.keys():
            self.dict_all_predictions[key] = pd.DataFrame()


    def make_prediction(self, df_out_sample, model):
        
        df_pred = df_out_sample.copy()
        df_pred[self.col_to_pred + '_pred'] = model.predict(df_out_sample[self.modeling_features])
        return df_pred
    
    def get_n_days_rolling(self):
        if self.rolling_frq == '1M':
            return 21
        elif self.rolling_frq == '1W':
            return 7
        elif self.rolling_frq == '1D':
            return 1
        else:
            print('rolling_frq not supported: ' + self.rolling_frq)

    
    def alpha_estimation(self, df_r, alpha_estimation_method):
        
        if alpha_estimation_method == "Lasso": 
            model = Lasso(max_iter=int(1e5))
            
        elif alpha_estimation_method == "RollingOLS":
            model = LinearRegression(normalize=True)
            
        elif alpha_estimation_method == "Ridge":
            model = Ridge()
            
        elif alpha_estimation_method == "ElasticNet":
            model = ElasticNet(max_iter=100000)
            
        elif alpha_estimation_method == "xgboost":
 
            model = xgb.XGBRegressor(random_state = 42, n_jobs = -1, verbosity = 0)
            
        elif alpha_estimation_method == "random_forest": 
            model = RandomForestRegressor(random_state=0 , n_jobs=-1)
            
        # Fit the model
        model.fit(df_r[self.modeling_features], df_r[self.col_to_pred])
        
        return model
    
    def run_backtest(self):
        
        self.df["constant"] = 1
    
        for dt in tqdm(pd.date_range(start=self.df['datetime'].min() + pd.Timedelta(days=self.look_back_prm),
                                end=self.df['datetime'].max() - pd.Timedelta(days=self.get_n_days_rolling()+1), 
                                freq=self.rolling_frq)):
            
        
            for cfg_name, cfg in self.configurations.items():
                ### ASSIGN CONFIGURATION PARAMETERS #########
                #############################################
                df = self.df 

                # step1: restrict dataset to insample
                df_r = df.loc[np.logical_and(
                    df['datetime'] >= dt - pd.Timedelta(days=self.look_back_prm), 
                    df['datetime'] < dt ), :].copy()
                
                
                # step3: run alpha estimation method
                model = self.alpha_estimation(df_r, cfg['alpha_estimation_method'])

                # step4: set out of sample period 
                df_out_sample = df.loc[np.logical_and(
                    df['datetime'] >= dt, 
                    df['datetime'] < dt + pd.Timedelta(days=self.get_n_days_rolling())), :].copy()
                
                
                if df_out_sample.empty:
                    continue
                
                # step6: make prediction for out of sample
                df_pred = self.make_prediction(df_out_sample, model)
                
                # save all predictions of one specific configuration to a dictionary
                self.dict_all_predictions[cfg_name] = pd.concat([self.dict_all_predictions[cfg_name], df_pred])
        
                # save the feature importance if the model is random forest or xgboost
                if cfg['alpha_estimation_method'] in ['xgboost', 'random_forest']:
                    key = str(list(df_out_sample['datetime'].unique())[0])
                    # saving the feature importance for insample data in a dictionary
                    self.dict_feature_importance[cfg['alpha_estimation_method']][key] = dict(zip(self.modeling_features, list(model.feature_importances_)))    