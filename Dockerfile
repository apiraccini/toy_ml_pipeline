FROM apache/airflow:2.6.0
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt