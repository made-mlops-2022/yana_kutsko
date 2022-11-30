FROM python:3.8

COPY requirements.txt make_predictions.py /

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "make_predictions.py"]