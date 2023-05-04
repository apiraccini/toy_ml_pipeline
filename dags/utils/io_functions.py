import pandas as pd
import os.path

def save_files(df_list: list[pd.DataFrame]):
    '''
    saves each dataframe as csv to the /data folder (volume, so temporary)
    the file name corresponds to the dataframe 'name' attribute
    '''
    
    [df.to_csv(f'/opt/airflow/data/{df.name}.csv') for df in df_list]


def load_files(names_list: list[str]):
    '''
    load each csv file in the list of names from /data folder (volume)
    returns a list of loaded dataframes
    '''

    out = [pd.read_csv('/opt/airflow/data/{name}.csv') for name in names_list if os.path.isfile('/opt/airflow/data/{name}.csv')]
    return out