# ROCm 6.4 Docker base image
FROM rocm/dev-ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app/

RUN apt-get update && sudo apt-get install -y hiprand hipblas rocrand rocsolver rocsparse

# Install the application dependencies
RUN pip install jupyterlab plotly rich yfinance tqdm
RUN pip install amd-hipdf==1.0.0b1 --extra-index-url=https://pypi.amd.com/simple

CMD ["/bin/bash"]