FROM registry.access.redhat.com/ubi9/ubi

RUN mkdir /app

RUN dnf -y update \
    && dnf -y install python3.11 python3.11-devel python3.11-pip ca-certificates \
    && dnf clean all

WORKDIR /app

COPY requirements.txt /app

RUN python3.11 -m pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

ENTRYPOINT ["python3.11", "runner.py"]
