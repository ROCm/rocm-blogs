FROM rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch

# Prevent prompts during session
ARG DEBIAN_FRONTEND=noninteractive
 
ENV CUDA_VISIBLE_DEVICES=0
ENV ROCM_VERSION=6.3

RUN pip install --upgrade pip
RUN pip install jupyter lab 
RUN pip install transformers accelerate datasets lm_eval tokenicer device-smi colorlog sentencepiece logbar

# Install GPTQModel from source
RUN git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel && pip install -v . --no-build-isolation

WORKDIR /usr/src/app/ 

# Default command to run the container
CMD ["/bin/bash"]
