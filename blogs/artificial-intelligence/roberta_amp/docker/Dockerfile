FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN pip install --upgrade pandas
RUN pip install --upgrade transformers
RUN pip install --upgrade accelerate
RUN pip install --upgrade datasets
RUN pip install --upgrade ipywidgets
RUN pip install plotly

WORKDIR /usr/src/app
