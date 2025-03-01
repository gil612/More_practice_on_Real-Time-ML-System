# Build stage
FROM python:3.12-slim as builder

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/ && \
    ldconfig

# Set environment variables
ENV POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    TA_INCLUDE_PATH=/usr/include \
    TA_LIBRARY_PATH=/usr/lib \
    PYTHONPATH=/app

# Install Poetry and basic dependencies
RUN pip install --no-cache-dir \
    poetry==${POETRY_VERSION} \
    numpy \
    wheel \
    setuptools \
    Cython

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN poetry lock && \
    poetry install --no-interaction --no-ansi --no-root --no-cache && \
    rm -rf /root/.cache/pip/* /root/.cache/poetry/*

# Copy the application code
COPY . .

# Final stage
FROM python:3.12-slim

# Copy necessary files from builder
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Update shared library cache
RUN ldconfig

WORKDIR /app

# Copy application code
COPY . .

# Command to run the application
CMD ["python", "-m", "src.training"]