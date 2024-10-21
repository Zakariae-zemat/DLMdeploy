# fastapi_app/app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

app = FastAPI()

# Load the trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load('fastap/mnist_ann-model.pth'))
model.eval()

model.eval()
class PredictionRequest(BaseModel):
    image: bytes

# Preprocess the image for prediction
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# API route for predicting the number
@app.post("/predict/")
async def get_number_prediction(image: UploadFile = File(...)):
    image_data = await image.read()
    img = Image.open(BytesIO(image_data))
    preprocessed_image = preprocess_image(img)
    
    # Make prediction
    with torch.no_grad():
        predictions = model(preprocessed_image)
    predicted_digit = torch.argmax(predictions, dim=1).item()
    
    return {"prediction": int(predicted_digit)}


@app.get("/")
async def getmessage():
    
    return {"prediction":"donne"}