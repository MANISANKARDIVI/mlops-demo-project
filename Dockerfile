FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python3", "app.py"]
