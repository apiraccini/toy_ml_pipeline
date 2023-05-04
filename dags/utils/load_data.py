import pandas as pd
import datetime as dt
import yfinance as yf

from utils.io_functions import save_files
import utils.config as cfg

def load_data():
    '''
    download stock price data from yahoo finance and save
    company is set in config file (default GOOG) 
    data range from today until 10 years ago
    '''
    # set date points
    today = dt.datetime.now()
    start = today - dt.timedelta(days=365*10)

    # dowload stock price data
    df = yf.download(cfg['company'], start = start.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    df = df[['Close']].rename(columns={'Close': 'y'})

    # save data
    df.name = f'data_raw'
    save_files([df])