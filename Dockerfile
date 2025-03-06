FROM rust:1.73-slim as rust-builder

WORKDIR /app/rust_8bit
COPY rust_8bit/ .
RUN cargo build --release

FROM python:3.9-slim

# Install Rust
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy Rust project
COPY rust_8bit/ /app/rust_8bit/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build Rust module
RUN cd /app/rust_8bit && maturin develop

# Copy Python app
COPY app.py generate.py ./
COPY templates/ ./templates/
COPY static/ ./static/

CMD gunicorn --bind 0.0.0.0:$PORT app:app