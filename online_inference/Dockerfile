FROM python:3.8-slim

COPY requirements.txt ./online_inference/requirements.txt

WORKDIR ./online_inference

RUN pip install --upgrade -r requirements.txt

COPY client.py generate_test_data.py main.py model.pkl model_data.py run.sh test_main.py ./

EXPOSE 3000

CMD ["bash", "run.sh"]