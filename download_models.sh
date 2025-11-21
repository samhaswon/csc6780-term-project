#!/usr/bin/env bash

echo "Downloading Models to ./models/"
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/birefnet.onnx -O models/birefnet.onnx
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/birefnet.pth -O models/birefnet.pth
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/u2net.onnx -O models/u2net.onnx
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/u2net.pth -O models/u2net.pth
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/u2netp.onnx -O models/u2netp.onnx
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/u2netp.pth -O models/u2netp.pth
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/u2netp_chunks.onnx -O models/u2netp_chunks.onnx
wget https://github.com/samhaswon/csc6780-term-project/releases/download/v0.0.0/u2netp_chunks.pth -O models/u2netp_chunks.pth
echo "Download complete"
