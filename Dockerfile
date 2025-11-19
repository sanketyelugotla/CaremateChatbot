# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose and set the default port (Render will override PORT env variable)
ENV PORT=8000
EXPOSE 8000

# Command to run the Flask app
CMD ["python", "app.py"]