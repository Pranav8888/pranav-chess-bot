FROM python:3.12-slim

# Install Stockfish with root privileges in the build environment
RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining project files
COPY . .

# Run application binding to Render's assigned PORT variable
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app