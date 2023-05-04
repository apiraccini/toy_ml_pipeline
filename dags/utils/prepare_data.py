import pandas as pd
import datetime as dt

from utils.io_functions import save_files, load_files
import utils.config as cfg

def window_df(data: pd.DataFrame, past: int, future: int) -> pd.DataFrame:
    '''
    shift and name values in order to have past days of observations as features 
    and future days as targets in a preprocessed dataframe
    '''

    df = data.copy()
    for i in range(0, past):
        df[f'y_t-{i}'] = df['y'].shift(i)
    for j in range(1, future+1):
        df[f'y_t+{j}'] = df['y'].shift(-j)

    df.drop(columns=['y'], inplace=True)
    df.rename(columns={'y_t-0':'y_t'}, inplace=True)
    df.dropna(inplace=True)
    
    return df

def prepare_data():
    '''
    process data obtaining lagged inputs and future values as targets
    then split into train and test set and save to local volume
    '''

    # load raw data and apply window function
    df = load_files(['data_raw'])[0]
    proc_df = window_df(df, past=cfg['past'], future=cfg['future'])

    # get feature and target names
    x_cols = [col for col in proc_df.columns if col.startswith('y_t-') or col=='y_t']
    y_cols = [col for col in proc_df.columns if col.startswith('y_t+')]

    # pop out last 10 observations as test set
    x = proc_df[x_cols][:-10]
    y = proc_df[y_cols][:-10]

    x_test = proc_df[x_cols][-10:]
    y_test = proc_df[y_cols][-10:]

    # save data
    proc_df.name = 'data_processed'
    x.name = 'x_train'
    x_test.name = 'x_test'
    y.name = 'y_train'
    y_test.name = 'y_test'
    save_files([proc_df, x, y, x_test, y_test])