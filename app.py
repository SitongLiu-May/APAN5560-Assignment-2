from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
from cnn_model import CNNModel
import io

app = FastAPI(title="CNN Image Classifier API")

model = CNNModel()
model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.get("/")
def read_root():
    return {"message": "Welcome to the CNN Classifier API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return {"predicted_class": int(predicted)}
