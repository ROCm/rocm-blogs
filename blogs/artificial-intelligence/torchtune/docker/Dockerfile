FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip
RUN pip install --upgrade ipywidgets
RUN pip install numpy==1.24.4
RUN pip install torchtune==0.2.1
RUN pip install jupyterlab
RUN pip install plotly

WORKDIR /usr/src/app
