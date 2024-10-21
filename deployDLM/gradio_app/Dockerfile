# Use a slim Python image to reduce the image size
FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the Gradio default port
EXPOSE 8080

# Run the Gradio app
CMD ["python", "interface.py"]
