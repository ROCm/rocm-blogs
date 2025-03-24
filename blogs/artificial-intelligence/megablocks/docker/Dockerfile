FROM rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Install the application dependencies
RUN pip install jupyterlab
RUN pip install regex
RUN pip install nltk
RUN pip install pybind11

RUN git clone --recurse-submodules -j8 https://github.com/ROCm/megablocks && \
    cd megablocks && \
    ./patch_torch.sh && \
    python setup.py install

RUN cd megablocks/third_party/Stanford-Megatron-LM && \
    git checkout rocm_6_3_patch && \
    ./apply_patch.sh

CMD ["/bin/bash"]