FROM python:3.12

# Set up the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code into the container
COPY fastapi_server.py .
COPY mnist_pytorch_model.pth .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

