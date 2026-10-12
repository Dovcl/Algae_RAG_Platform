FROM python:3.11-slim
LABEL project=algae-rag

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

COPY alembic.ini .
COPY alembic/ alembic/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]