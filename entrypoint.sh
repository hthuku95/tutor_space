#!/bin/bash

# Function to wait for PostgreSQL
wait_for_postgres() {
    if [ "$DB_ENGINE" = "postgresql" ]; then
        echo "Waiting for PostgreSQL..."
        while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
            sleep 0.1
        done
        echo "PostgreSQL started"
    fi
}

# Wait for database to be ready
wait_for_postgres

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start server
echo "Starting server..."
exec "$@"