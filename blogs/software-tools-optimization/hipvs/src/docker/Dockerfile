# ROCm7 + pytorch2.8 base image
FROM rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install -y libopenblas-dev
WORKDIR /usr/src/app/
# Install additional dependencies into the same venv where pytorch is installed.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install hipvs and the application dependencies
RUN pip install amd-hipvs==0.1.0 --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple/
RUN pip install amd-cupy==13.5.1 --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple/
RUN pip install ipywidgets
RUN pip install jupyterlab
RUN pip install adjustText h5py matplotlib
RUN pip install sentence_transformers

CMD ["/bin/bash"]
