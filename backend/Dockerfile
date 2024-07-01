FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn server:app --reload"]