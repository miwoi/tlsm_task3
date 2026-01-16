
FROM python:3.12-slim


RUN apt-get update && apt-get install -y \
    git \
    python3-tk \
    x11-apps \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install uv


CMD ["sleep", "infinity"]