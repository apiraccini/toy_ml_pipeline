import numpy as np
import pandas as pd
import datetime as dt
import joblib

from sklearn.multioutput import RegressorChain
from sklearn.ensemble import HistGradientBoostingRegressor

from utils.io_functions import load_files, save_files
import utils.config as cfg

def fit_best_model():
    '''fit best model on all data and save it'''

    # load data and experiment info
    proc_df, exp_df = load_files(['data_processed', 'experiment_info'])
    
    # get feature and target names
    x_cols = [col for col in proc_df.columns if col.startswith('y_t-') or col=='y_t']
    y_cols = [col for col in proc_df.columns if col.startswith('y_t+')]

    x = proc_df[x_cols]
    y = proc_df[y_cols]

    # fit model with best hyperparameters on all data
    best_params = exp_df[exp_df['params_names'].values[0]].to_dict(orient='records')[0]
    model = RegressorChain(base_estimator=HistGradientBoostingRegressor(**best_params))
    model.fit(x, y)

    # save model
    now = dt.now().strftime('%d-%m-%Y_%H:%M:%S')
    filename = f'model_{cfg["company"]}_{now}.pkl'
    joblib.dump(model, f'/opt/airflow/models/{filename}', compress=1)