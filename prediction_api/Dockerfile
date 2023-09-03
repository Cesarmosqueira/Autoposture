FROM python:3.8-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py /app
COPY src /app/src

COPY .env.example .env

RUN pip install python-dotenv && python -c "import dotenv; dotenv.load_dotenv('.env')"

EXPOSE 8080
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
