# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "front.py", "--server.port=8501", "--server.address=0.0.0.0"]
