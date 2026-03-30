import io
import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

MODEL_PATH = "/models/resnet18_mnist.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ResNet18 MNIST Inference API")

# 학습 때와 동일한 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307, 0.1307, 0.1307),
        (0.3081, 0.3081, 0.3081)
    )
])


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/")
def root():
    return {
        "message": "ResNet18 MNIST inference server is running",
        "device": str(device),
        "model_path": MODEL_PATH
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if os.path.exists(MODEL_PATH) else "model_missing",
        "model_exists": os.path.exists(MODEL_PATH),
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")

        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(torch.max(probs).item())

        return {
            "prediction": pred,
            "confidence": round(confidence, 6)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"추론 실패: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=False)
