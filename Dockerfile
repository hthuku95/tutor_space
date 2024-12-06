# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Install core Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir \
    Django==5.1.1 \
    dj-database-url==2.1.0 \
    psycopg2-binary==2.9.10 \
    python-dotenv==1.0.1

# Install rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy project files
COPY . .

# Create and set permissions for media and static directories
RUN mkdir -p /app/media /app/static \
    && chmod -R 755 /app/media /app/static

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
