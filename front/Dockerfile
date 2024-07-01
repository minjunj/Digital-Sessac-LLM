FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install streamlit

EXPOSE 8501

CMD ["streamlit", "run", "front.py", "--server.port=8501", "--server.address=0.0.0.0"]
