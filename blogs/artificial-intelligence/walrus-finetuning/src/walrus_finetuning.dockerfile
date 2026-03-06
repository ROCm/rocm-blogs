FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0

WORKDIR /workspace

# Install vim, unzip, and ffmpeg
RUN apt-get update && apt-get install -y vim unzip ffmpeg

# Install The Well (dataset library)
RUN pip install the_well

# Clone & install Walrus
RUN git clone https://github.com/PolymathicAI/walrus.git && \
    cd walrus && \ 
    pip install -e . --no-deps

# Copy configs and requirements
COPY ./configs ./walrus/walrus/configs

COPY ./run_scripts ./walrus/walrus/run_scripts

COPY ./requirements.txt ./walrus/requirements.txt

# Install additional Python dependencies
RUN pip install -r ./walrus/requirements.txt