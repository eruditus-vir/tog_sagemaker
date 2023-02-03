FROM python:3.7.16

ARG APP_PATH=/app

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY tog_sagemaker /opt/code
WORKDIR /opt/code
RUN pip install --upgrade pip
RUN pip install -r requirements.txt