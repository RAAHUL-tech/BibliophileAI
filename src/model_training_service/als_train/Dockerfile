FROM python:3.10
WORKDIR /app
COPY als_train.py .   
COPY requirements.txt .  
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["python", "als_train.py"]
