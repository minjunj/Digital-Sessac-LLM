#FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

FROM python:3.10

COPY . .

#RUN pip install -r requirements.txt

RUN pip install transformers

RUN pip install torch

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn server:app --reload"]