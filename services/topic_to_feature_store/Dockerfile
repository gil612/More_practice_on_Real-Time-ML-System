# Let's start with a python 3.12 slim image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python Poetry with version 1.8.3
RUN pip install poetry==1.8.3

# Copy the source code to the working directory
COPY . /app

# Install the dependencies
RUN poetry install

# Run the application
CMD ["poetry", "run", "python", "src/main.py"]