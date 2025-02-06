# Let's start with a python 3.12 slim image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y iputils-ping net-tools \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python Poetry with version 1.8.3
RUN pip install poetry==1.8.3

# Copy only pyproject.toml first to leverage Docker cache
COPY pyproject.toml ./

# Delete any existing poetry.lock file and generate a new one
RUN rm -f poetry.lock && \
    poetry lock && \
    poetry install --no-root

# Copy the rest of the source code
COPY . .

# Install the project itself with a fresh lock
RUN poetry install

# Run the application
CMD ["poetry", "run", "python", "src/main.py"]