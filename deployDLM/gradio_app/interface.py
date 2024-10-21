import gradio as gr
import requests
from PIL import Image
import imageio 
# API endpoint for FastAPI prediction
API_URL = "http://fastapi_container:8000/predict/"

def predict_digit(image):
    
    imageio.imwrite("temp.png", image)  # Save image in PNG format

    with open("temp.png", "rb") as f:
        response = requests.post(API_URL, files={"image": f})
    prediction = response.json()["prediction"]
    return f"The model predicted: {prediction}"

# Create Gradio Interface
iface = gr.Interface(fn=predict_digit, inputs="image", outputs="text")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8088)
