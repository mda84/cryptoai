# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required by some packages (git, ffmpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Upgrade pip and install project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit dashboard
CMD ["streamlit", "run", "main.py", "--server.enableCORS", "false"]
