FROM apache/airflow:2.6.0
RUN pip install --upgrade pip
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt