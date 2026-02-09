FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY als_train.py .
ENTRYPOINT ["./entrypoint.sh"]
