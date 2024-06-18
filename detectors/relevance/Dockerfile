FROM registry.access.redhat.com/ubi9/ubi

RUN mkdir /app

RUN dnf -y update  \
    && dnf -y install python3.11 python3.11-devel python3.11-pip ca-certificates procps \
    && dnf clean all

WORKDIR /app

COPY requirements.txt /app

RUN python3.11 -m pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

EXPOSE 80

ENV WORKERS 1

CMD uvicorn main:app --host 0.0.0.0 --port 80  --workers $WORKERS
