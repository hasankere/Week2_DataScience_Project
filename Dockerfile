
    # Use an official Python runtime as the base image
    FROM python:3.9-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy requirements and install them
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # Copy the rest of the application code
    COPY . .

    # Command to run your application
    CMD ["python", "-m", "src"]
    