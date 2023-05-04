cfg = {
    'db_engine': 'postgresql+psycopg2://airflow:airflow@postgres/airflow',
    'db_schema': 'public',
    'db_experiments_table': 'experiments',
    'db_batch_table': 'batch_data',
    'cv_folds': 4,
    'n_jobs': 4,
    'seed': 42,
    'company':'GOOG',
    'future': 7,
    'past': 30
}