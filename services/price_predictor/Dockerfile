# Let's start with a python 3.10 slim image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install build-essential which includes gcc and other build tools
RUN apt-get update && apt-get install -y build-essential wget

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Install Python Poetry with version 1.8.3
RUN pip install poetry==1.8.3

# Copy the source code to the working directory
COPY . /app

# Install the dependencies
RUN poetry install

# Run the application
CMD ["poetry", "run", "python", "src/training.py"]