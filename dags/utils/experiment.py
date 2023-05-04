import numpy as np
import pandas as pd
import datetime as dt

from sklearn.multioutput import RegressorChain
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

import optuna

from utils.io_functions import load_files, save_files
import utils.config as cfg

def objective(trial, x, y, cv, fixed_params):
    '''objective function for optimization'''
    
    # trial parameters
    tuning_params = {
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
    }
    params = {**fixed_params, **tuning_params}

    # train and score with cv
    model = RegressorChain(base_estimator=HistGradientBoostingRegressor(**params))
    cv_res = cross_validate(
        estimator = model, 
        X = x, 
        y = y,
        scoring = {
            'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False, multioutput='uniform_average'),
            #'mae': make_scorer(mean_absolute_error, greater_is_better=False, multioutput='uniform_average'),
            #'rmse': make_scorer(mean_squared_error, greater_is_better=False, multioutput='uniform_average', squared=False)
        },
        cv = cv,
        n_jobs= cfg['n_jobs']
    )

    # return mean cv score 
    return -np.mean(cv_res['test_mape'])

def experiment():
    '''
    find best set of hyperparameters via cv on train set
    then score predictions on test set and save info
    '''

    x, x_test, y, y_test = load_files(['x_train', 'x_test', 'y_train', 'y_test'])

    # define cross validation strategy
    cv = KFold(n_splits = cfg['cv_folds'], shuffle = True, random_state = cfg['seed'])

    # fixed params
    fixed_params = {
        'loss': 'squared_error',
        'scoring': 'neg_mean_absolute_percentage_error',
        'verbose': 0,
        'early_stopping': True,
        'validation_fraction': .1,
        'n_iter_no_change': 15,
        'random_state': 42 
    }

    # create study
    sampler = optuna.samplers.TPESampler(seed=cfg['seed'])
    max_trials = 1
    time_limit = 3600 * 0.5

    study = optuna.create_study(
        sampler=sampler,
        study_name= f"{cfg['modelname']}_optimization",
        direction='minimize')

    # perform optimization
    print(f"Starting {cfg['modelname']} optimization...")
    study.optimize(
        func = lambda trial: objective(trial, x, y, cv, fixed_params),
        n_trials = max_trials,
        timeout = time_limit,
    )

    # optimization results
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best score: {study.best_value}")

    best_params = {**fixed_params, **study.best_trial.params}
    print("Best trial parameters:")
    for k, v in best_params.items():
        print(f"\t{k}: {v}")

    # test set scoring
    model = HistGradientBoostingRegressor(**best_params)
    model.fit(x, y)
    y_pred = model.predict(x_test)

    mape = mean_absolute_percentage_error(y_pred, y_test, multioutput='uniform_average')
    mae = mean_absolute_error(y_pred, y_test, multioutput='uniform_average')
    rmse = mean_squared_error(y_pred, y_test, multioutput='uniform_average', squared=False)

    # save results
    now = dt.now().strftime("%d-%m-%Y_%H:%M:%S")
    cfg_df = pd.DataFrame({'date': now, 'params_names': list(best_params.keys())})
    params_df = pd.DataFrame(best_params)
    test_df = pd.DataFrame({'rmse':rmse, 'mae':mae, 'mape':mape})
    
    exp_df = pd.concat([cfg_df, params_df, test_df], axis=1)
    exp_df.name = f'experiment_log'
    save_files([exp_df])
