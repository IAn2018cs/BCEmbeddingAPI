FROM nvidia/cuda:12.5.0-base-ubuntu22.04

WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装基本依赖和Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 确保使用Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# 创建并激活虚拟环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# 验证Python版本
RUN python3 --version

# 安装PyTorch，兼容CUDA 12.5
RUN pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 复制requirements.txt并安装依赖
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "300", "--log-level", "info"]