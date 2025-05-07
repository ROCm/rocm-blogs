# ROCm 6.4 Docker base image
FROM rocm/dev-ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app/

# Install the application dependencies
RUN pip install jupyterlab plotly rich
RUN pip install ipykernel==7.0.0a1
RUN pip install amd-hipdf==1.0.0b1 --extra-index-url=https://pypi.amd.com/simple

CMD ["/bin/bash"]