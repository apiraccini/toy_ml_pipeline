import pandas as pd
from sqlalchemy import create_engine

from utils.io_functions import load_files
import utils.config as cfg

db_engine = cfg.params["db_engine"]
db_schema = cfg.params["db_schema"]
table_batch = cfg.params["db_batch_table"] 

def save_batch_data():
    '''append experiment log to sql database'''

    df = load_files(['experiment_log'])[0]
    engine = create_engine(db_engine)
    df.to_sql(table_batch, engine, schema=db_schema, if_exists='append', index=False)