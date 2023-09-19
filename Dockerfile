FROM python:3.9.6-slim

WORKDIR /api


COPY SimpleHTR/ /api
COPY api.py /api/
COPY requirements.txt /api/
COPY requirements-ppocr.txt /api/

RUN apt-get update -y \
    && apt-get install build-essential -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install flit \
    && FLIT_ROOT_INSTALL=1 flit install --deps production \
    && rm -rf $(pip cache dir)

RUN pip install --upgrade pip \
    pip install -r requirements.txt \
    pip install -r requirements-ppocr.txt

CMD python api.py
