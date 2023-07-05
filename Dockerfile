FROM waggle/plugin-base:1.1.1-ml

# cache model weights at build time
ADD https://download.pytorch.org/models/resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
ADD https://download.pytorch.org/models/resnet34-b627a593.pth /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
ADD https://download.pytorch.org/models/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
ADD https://download.pytorch.org/models/resnet101-63fe2227.pth /root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
