FROM python:3.11-slim

WORKDIR /opt/program

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py serve.py ./

EXPOSE 8080

ENTRYPOINT ["python", "serve.py"]