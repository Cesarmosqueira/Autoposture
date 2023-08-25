FROM python:3.8-slim

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

WORKDIR /app

CMD ["python", "app.py"]

