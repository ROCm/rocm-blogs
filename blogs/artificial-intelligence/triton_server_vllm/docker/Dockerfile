# tritonserver is the name of the image built from source
FROM tritonserver 

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip
RUN pip install transformers==4.44.2
RUN pip install jupyterlab
RUN pip install requests

WORKDIR /usr/src/app