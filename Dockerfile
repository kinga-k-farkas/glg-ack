FROM python:3.9-slim

WORKDIR /app

COPY ./models models
COPY ./src/deployment src/deployment
COPY ./setup.py ./
COPY ./requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app/src/deployment

CMD ["/bin/sh", "-c", "streamlit run app.py --browser.serverAddress 0.0.0.0 --server.port 80"]