# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Expose and set the default port (Render will override PORT env variable)
ENV PORT=8000
EXPOSE 8000

# Copy a small start script and make it executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Healthcheck for readiness
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Command to run the Flask app via the start script (uses shell so $PORT expands)
CMD ["/bin/sh", "/app/start.sh"]