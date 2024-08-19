FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.9_pytorch_2.0.1

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip
RUN pip install --upgrade transformers
RUN pip install --upgrade scipy
RUN pip install --upgrade ipywidgets
RUN pip install --upgrade numba
RUN pip install plotly
RUN pip install gluonts
RUN pip install ujson
RUN pip install datasets
RUN pip install evaluate
RUN pip install accelerate
RUN pip install pandas==1.5.3
RUN pip install jupyterlab

WORKDIR /usr/src/app
