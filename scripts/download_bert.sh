#!/bin/bash
# Download bert-base-uncased from HuggingFace
# Usage: bash scripts/download_bert.sh

MODEL_DIR="models/bert"
mkdir -p "$MODEL_DIR"

echo "Downloading bert-base-uncased to $MODEL_DIR/ ..."

# Config
wget -O "$MODEL_DIR/config.json" \
  "https://huggingface.co/bert-base-uncased/resolve/main/config.json"

# Model weights (440MB)
wget -O "$MODEL_DIR/pytorch_model.bin" \
  "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"

# Tokenizer files
wget -O "$MODEL_DIR/vocab.txt" \
  "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"

wget -O "$MODEL_DIR/tokenizer.json" \
  "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json"

# Optional but good to have
wget -O "$MODEL_DIR/tokenizer_config.json" \
  "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json"

echo "Done! Files saved to $MODEL_DIR/"
ls -lh "$MODEL_DIR/"
