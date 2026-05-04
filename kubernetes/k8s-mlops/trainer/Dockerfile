FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# torchvision 설치 (베이스 이미지에 없을 수 있음)
RUN pip install --no-cache-dir torchvision

# 학습 코드 복사
COPY train.py .

# 기본 실행 명령
CMD ["python", "train.py"]
