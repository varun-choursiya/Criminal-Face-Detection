FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir -p /app/uploads

EXPOSE 5000

ENV FLASK_ENV=production
ENV FLASK_DEBUG=0

CMD ["gunicorn", "main:app", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5000"]
