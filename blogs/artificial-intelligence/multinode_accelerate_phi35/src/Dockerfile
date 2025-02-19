FROM rocm/pytorch:rocm6.2.1_ubuntu20.04_py3.9_pytorch_release_2.3.0

ARG DEBIAN_FRONTEND=noninteractive #Suppress interactive prompts during package installation

RUN pip install --upgrade pip
RUN pip install numpy==1.24.4
RUN pip install scipy==1.13.1
RUN pip install transformers==4.45.1
RUN pip install accelerate==0.34.2
RUN pip install datasets==3.0.1

WORKDIR /usr/src/app