FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY linucb_train.py .
COPY helper.py .

ENTRYPOINT ["./entrypoint.sh"]