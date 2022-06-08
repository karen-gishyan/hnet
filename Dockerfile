FROM python:3.8-slim-buster
WORKDIR /hnet
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
