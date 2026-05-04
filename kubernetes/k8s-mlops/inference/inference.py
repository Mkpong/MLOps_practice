"""
ResNet18 MNIST 추론 서비스 (FastAPI)
- /models/resnet18_mnist.pt 에서 모델 로드
- POST /predict 로 이미지 받아 분류 결과 반환
"""
import io
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import uvicorn

# ===== 설정 =====
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/resnet18_mnist.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ResNet18 MNIST Inference")

# 학습 시와 동일한 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===== 모델 로드 =====
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.to(device).eval()
    return m


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    print(f"Model loaded on {device}: {MODEL_PATH}", flush=True)


# ===== 엔드포인트 =====
@app.get("/")
def root():
    return {
        "message": "ResNet18 MNIST inference server is running",
        "device": str(device),
        "model_path": MODEL_PATH,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": os.path.exists(MODEL_PATH),
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # MNIST → 흑백
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, prediction = probs.max(1)

        return {
            "prediction": int(prediction.item()),
            "confidence": float(confidence.item()),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"예측 실패: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
