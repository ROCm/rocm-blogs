# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
RUN apk add --no-cache gcc musl-dev linux-headers
COPY fastapi-requirements.txt fastapi-requirements.txt
RUN pip install -r fastapi-requirements.txt
EXPOSE 8000
COPY app.py app.py
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0"]