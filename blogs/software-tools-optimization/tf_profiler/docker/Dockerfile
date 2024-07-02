FROM rocm/tensorflow:rocm6.0-tf2.12-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install jupyterlab
RUN pip install -U tensorflow-datasets
RUN pip install matplotlib
RUN pip install -U tensorboard_plugin_profile

WORKDIR /usr/src/app
