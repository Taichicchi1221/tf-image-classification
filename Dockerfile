FROM gcr.io/kaggle-gpu-images/python:latest
RUN apt -y remove nvidia-*
RUN pip install pip --upgrade \
    && pip install mlflow --upgrade \
    && pip install hydra-core --upgrade

