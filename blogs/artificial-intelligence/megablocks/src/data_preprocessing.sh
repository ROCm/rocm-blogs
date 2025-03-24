#!/bin/bash

BASE_DATA_PATH=/megablocks/third_party/Stanford-Megatron-LM/tools
cd ${BASE_DATA_PATH}

echo "Obtaining dataset, vocabulary, and merge table from HuggingFace:"

wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d oscar-1GB.jsonl.xz
wget --output-document=gpt2-vocab.json https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
wget --output-document=gpt2-merges.txt https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt

echo "Preprocessing training data:"

python preprocess_data.py \
    --input ${BASE_DATA_PATH}/oscar-1GB.jsonl \
    --output-prefix ${BASE_DATA_PATH}/my-gpt2 \
    --vocab-file ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod \
    --workers 8 \
    --chunk-size 10
cd ..