FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install torch torchvision fastapi uvicorn pillow python-multipart


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
