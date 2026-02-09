FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY sasrec_model.py .
COPY sasrec_data.py .
COPY sasrec_train.py .

ENTRYPOINT ["./entrypoint.sh"]
