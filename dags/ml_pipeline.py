from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

from utils import load_data
from utils import prepare_data
from utils import save_batch_data
from utils import experiment
from utils import track_experiment_log
from utils import fit_best_model

default_args= {
    'owner': 'Alessio Piraccini',
    'email_on_failure': False,
    'email': ['alepiracci@gmail.com'],
    'start_date': datetime(2023, 4, 5)
}

with DAG(
    "ml_pipeline",
    description='End-to-end toy ml pipeline',
    schedule_interval='@daily',
    default_args=default_args, 
    catchup=False
    ) as dag:

    # task 1
    with TaskGroup('create_storage_structures') as create_storage_structures:

        # 1.1
        create_experiment_tracking_table = PostgresOperator(
            task_id="create_experiment_tracking_table",
            postgres_conn_id='postgres_default',
            sql='sql/create_experiments_table.sql'
        )

        # 1.2
        create_batch_data_table = PostgresOperator(
            task_id="create_batch_data_table",
            postgres_conn_id='postgres_default',
            sql='sql/create_batch_data_table.sql'
        )

    # task 2
    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data
        )
    
    # task 3
    with TaskGroup('prepare_data') as prepare_data:

        # 3.1
        preprocess_data = PythonOperator(
            task_id='preprocess_data',
            python_callable=prepare_data
        )

        # 3.2
        save_batch_data = PythonOperator(
            task_id='save_batch_data',
            python_callable=save_batch_data
        )

    # task 4
    with TaskGroup('experiment') as experiment:

        # 4.1
        hyperparam_tuning = PythonOperator(
            task_id='hyperparam_tuning',
            python_callable=experiment
        )

        # 4.2
        track_experiment_log = PythonOperator(
            task_id='track_experiment_log',
            python_callable=track_experiment_log
        )
        
    # task 5
    fit_best_model = PythonOperator(
        task_id='fit_best_model',
        python_callable=fit_best_model
    )

    create_storage_structures >> load_data >> prepare_data >> experiment >> fit_best_model