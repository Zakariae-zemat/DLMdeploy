FROM python:3.12

# Set up the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Gradio app code
COPY gradio_app.py .

# Expose the port Gradio runs on
EXPOSE 8088

# Run the Gradio app
CMD ["python", "gradio_app.py"]

