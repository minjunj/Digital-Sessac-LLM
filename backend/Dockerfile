FROM python:3.10

COPY . .

RUN pip install transformers

RUN pip install torch

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]