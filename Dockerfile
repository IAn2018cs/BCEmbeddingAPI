FROM nvidia/cuda:12.5.0-base-ubuntu20.04

WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装PyTorch，兼容CUDA 12.5
RUN pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 复制requirements.txt并安装依赖
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 5000

# 使用gunicorn启动应用，确保日志输出到标准输出
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "300", "--log-level", "info", "--capture-output"]