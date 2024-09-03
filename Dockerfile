ARG BASE=nvidia/cuda:12.4.0-base-ubuntu22.04
FROM ${BASE}


RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_for_docker.txt requirements_for_docker.txt
RUN pip install -r requirements_for_docker.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]