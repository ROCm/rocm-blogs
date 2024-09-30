FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip
RUN pip install --upgrade numba
RUN pip install --upgrade ipywidgets
RUN pip install jupyterlab
RUN pip install scipy==1.9.0
RUN pip install pandas==1.5.3
RUN pip install transformers==4.41.2
RUN pip install librosa
RUN pip install jiwer
RUN pip install datasets==2.21.0
RUN pip install accelerate==0.32.1
RUN pip install unidecode
RUN pip install evaluate
RUN pip install pyannote.audio==3.3.1
RUN pip install torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0


WORKDIR /usr/src/app
