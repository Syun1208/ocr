FROM python:3.8.2

ENV PYTHONUNBUFFERED True

WORKDIR /api


COPY ./ /api/


RUN apt-get update -y \
    && apt-get install build-essential -y \
    && apt-get update \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD python api.py