# ベースイメージ
FROM ubuntu:22.04

USER root

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    curl && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools

# ワークディレクトリを設定
WORKDIR /workspace

# 必要なPythonライブラリをインストール
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

ARG HF_USERNAME
ARG HF_TOKEN
ENV HF_USERNAME=${HF_USERNAME}
ENV HF_TOKEN=${HF_TOKEN}

RUN git clone https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm \
    /usr/local/lib/python3.10/dist-packages/style_bert_vits2/bert/deberta-v2-large-japanese-char-wwm