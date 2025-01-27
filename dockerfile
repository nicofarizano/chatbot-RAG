FROM python:3.9-slim

# Set environment variables
ENV USER_AGENT="promtior-bot"
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean

# Copy the application files
COPY . /app

# Upgrade pip and install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required port
EXPOSE 11435

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11435"]