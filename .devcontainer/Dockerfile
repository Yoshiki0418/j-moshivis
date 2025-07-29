# ベースイメージ
FROM ubuntu:22.04

USER root

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools

# ワークディレクトリを設定
WORKDIR /workspace

# 必要なPythonライブラリをインストール
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt