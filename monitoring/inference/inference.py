from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
from torchvision import models, transforms
from PIL import Image
import torch, io, uvicorn

app = FastAPI(title="ResNet18 Inference")    # /metrics 엔드포인트 자동 생성
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
tfm   = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

@app.get("/health")
async def health(): return {"status": "ok", "device": device}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad(): pred = model(x).argmax(1).item()
    return {"class": pred}
if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)

