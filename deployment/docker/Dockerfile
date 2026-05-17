# Use a lightweight Python base image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH to include the project root for module imports
ENV PYTHONPATH=/app

# Expose the port your FastAPI application runs on (from config.yaml)
EXPOSE 8000

# Command to run the application using uvicorn
# 'src.api.predict:app' refers to the FastAPI app instance in predict.py
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "src.api.predict:app", "--host", "0.0.0.0", "--port", "8000"]
