FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 cache-weights.py

ENTRYPOINT ["python3", "main.py"]
