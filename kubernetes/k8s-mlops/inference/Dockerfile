FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# 의존성 먼저 복사 (Docker 레이어 캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 추론 코드 복사
COPY inference.py .

EXPOSE 8000

CMD ["python", "inference.py"]
