"""
ResNet18 + MNIST 학습 스크립트
- Bind Mount된 /data에 데이터셋 다운로드
- Named Volume에 마운트된 /models에 학습된 모델 저장
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ===== 환경 설정 =====
DATA_DIR = "/data"
MODEL_DIR = "/models"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet18_mnist.pt")

EPOCHS = int(os.environ.get("EPOCHS", 2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
LR = float(os.environ.get("LR", 0.001))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

os.makedirs(MODEL_DIR, exist_ok=True)

# ===== 데이터 준비 =====
# MNIST는 1채널 28x28 → ResNet18은 3채널 224x224 입력 → 변환 필요
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ===== 모델 정의 =====
model = resnet18(weights=None)              # 사전학습 가중치 없이
model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST는 10 클래스
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== 학습 =====
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    train_acc = 100.0 * correct / total
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}%", flush=True)

# ===== 평가 =====
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

test_acc = 100.0 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%", flush=True)

# ===== 모델 저장 =====
torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved model to {MODEL_PATH}", flush=True)
