---
blogpost: true
blog_title: "Optimizing drug discovery tools on AMD MI300s Part 2: 3D Molecular Generation with SemlaFlow"
date: 3 Oct 2025
author: 'Vasumathi Neralla, Rui Sampaio'
thumbnail: 'thumbnail-semlaflow.jpeg'
tags: PyTorch, Optimization
category: Applications & models
target_audience: LifeScience developers
key_value_propositions: Steps to follow to run SemlaFlow on AMD MI300x, and also detailed optimization steps
language: English
myst:
    html_meta:
        "author": "Vasumathi Neralla, Rui Sampaio"
        "description lang=en": "Learn how to set up, run, and optimize SemlaFlow, a molecular generation tool, on AMD MI300X GPUs for faster drug discovery workflows"
        "keywords": "Life-science, Drug Discovery, 3D Molecular Generation, PyTorch, Docker, AI/ML"
        "vertical": "Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Design, Simulation & Modeling"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
        "amd_blog_authors": "Vasumathi Neralla, Rui Sampaio"
---

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# Optimizing Drug Discovery Tools on AMD MI300s Part 2: 3D Molecular Generation with SemlaFlow

This blog is part of a series of walkthroughs of Life Science AI models, stemming from [this article](https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html), which was a collaborative effort between AstraZeneca and AMD. The series delves into what was required in order to run drug discovery related AI workloads on AMD MI300X. The first post in this series, available [here](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html), focuses on REINVENT4, a molecular design tool used to generate and optimize candidate molecules. This blog, in particular, looks at SemlaFlow, an efficient 3D molecular generation model with latent attention and equivariant flow matching.

Simulation-based drug discovery pipelines are computational approaches used in the pharmaceutical industry to identify and optimize new drug candidates. These pipelines leverage computer simulations to model biological systems and predict how potential drug molecules will interact with their targets, such as proteins or nucleic acids.

3D molecular generation is a computational technique used to create three-dimensional representations of molecules, playing a crucial role in drug discovery. In the context of simulation-based drug discovery pipelines, 3D molecular generation is integrated with molecular modeling and virtual screening. By generating 3D structures, researchers can perform docking simulations to predict how well these molecules bind to target proteins. This step is critical in identifying potential drug candidates, as it provides insights into the binding affinity and specificity of the molecules. The ability to visualize and simulate these interactions in three dimensions enhances the accuracy and efficiency of the drug discovery process.

Previous approaches to 3D molecular generation often faced significant limitations, including very slow sampling times and the generation of molecules with poor chemical validity, hindering their practical application in drug discovery workflows.

SemlaFlow addresses these challenges by offering a state-of-the-art 3D molecular generation solution that provides a two order-of-magnitude speedup (equivalent to a more than 100-fold improvement) in sampling time compared to existing methods, requiring as few as 20 sampling steps. This efficiency is achieved through its novel and scalable E(3)-equivariant Semla architecture and its training with equivariant flow matching. SemlaFlow is also unique in its ability to generate a joint distribution over atom types, coordinates, bond types, and formal charges, providing a comprehensive molecular design without the need to infer parts of the distribution after generation.

In this blog, we show how minimal code changes can get you started, and we outline key steps to optimize SemlaFlow on AMD hardware. The environment used was a TensorWave node with 8 MI300Xs. Our primary focus here is training. While we use the prediction and evaluation tools to test models produced by the training utility, we do not concentrate on optimizing these tools. Our optimization efforts are dedicated solely to enhancing the efficiency of training the 3D molecular generation model.

## SemlaFlow code

SemlaFlow is a Linux-based application developed in Python, utilizing PyTorch for neural network implementations. Since PyTorch is compatible with ROCm, running SemlaFlow on AMD GPUs should be straightforward.

The original SemlaFlow repository contains four main scripts:

* `preprocess` - Used for preprocessing larger datasets into the internal representation used by the model for training
* `train` - Trains a MolFlow model on preprocessed data
* `evaluate` - Evaluates a trained model and prints the results
* `predict` - Runs the sampling for a trained model and saves the generated molecules

### Installation

Although the original SemlaFlow repository provides instructions for running it on Nvidia GPUs, it is relatively simple to run SemlaFlow on AMD GPUs. For instance, the mamba environment.yml file specifies a CUDA dependency: pytorch-cuda, which can be replaced with rocm-pytorch, the equivalent for AMD hardware. The remaining dependencies can stay unchanged as they are compute-agnostic packages.

To operate on a Kubernetes cluster, a dockerizable recipe is required. The rest of this blog assumes an environment where Docker is supported. Let's proceed.

### Dockerizing SemlaFlow

The simplest way to work with ROCm and PyTorch installations is to use a base image with these packages already installed. There are several different versions that can be found on [Docker Hub](https://hub.docker.com/r/rocm/pytorch/tags). The base image used was the latest at the time when we were running the experiments
[rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0](https://hub.docker.com/layers/rocm/pytorch/rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0/images/sha256-c76af9bfb1c25b0f40d4c29e8652105c57250bf018d23ff595b06bd79666fdd7)

Using an earlier version of the ROCm base image, caused some issues with SemlaFlow. Upgrading to the latest image with python3.12 fixed that issue.

So, the dockerfile basically uses the latest ROCm image as the base image and installs the packages that were listed in the mamba environment yml file using conda and pip.

A final step is to tell the image what to run when the container is created. We can provide an entrypoint script as follows:

```sh
#!/bin/bash

SCRIPT="$1"
OUTPUT_FILE="$2"
shift
shift

echo ${OUTPUT_FILE}
# Check if SCRIPT is one of the allowed values
if [[ "$SCRIPT" != "preprocess" && "$SCRIPT" != "train" && "$SCRIPT" != "evaluate" && "$SCRIPT" != "predict" ]]; then
  echo "Error: SCRIPT must be one of 'preprocess', 'train', 'evaluate', or 'predict'."
  exit 1
fi

python -m semlaflow."$SCRIPT" "$@" &> /output/${OUTPUT_FILE}

if [[ "$SCRIPT" == "train" ]]; then
  # Copy the model checkpoint to the output directory
  cp -r lightning_logs/version_* /output/
fi
```

It allows for multiple parameters to be passed, which script should be run, the output file and also the other arguments that are required to run the chosen script. The arguments necessary to run the scripts can be found in the scripts argument parser bit.

An example command:

```sh
bash entrypoint.sh rocm-semlaflow <script> <output-file> --data_path <path-to-dataset> <other_args>
````

Finally, this is the Dockerfile

```sh

FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0

RUN apt-get update -y \ 
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean

RUN conda install cxx-compiler tqdm wandb ipython certifi --channel conda-forge

RUN pip install numpy==1.26.2 pandas==2.2.2 scipy==1.11.4 rdkit lightning torchmetrics openbabel-wheel typing_extensions

RUN git clone https://github.com/rssrwn/semla-flow.git
WORKDIR semla-flow

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["", ""]
```

The above Dockerfile creates an image ready for SemlaFlow training, prediction and evaluation with proper drivers and packages.

* Ubuntu: 24.04
* Python: 3.12
* ROCm: 6.4.1
* PyTorch: 2.6.0
* Required packages to run SemlaFlow

Build the image using:

```sh
docker build -t rocm-semlaflow .
```

After the image is built, it can be run by first setting up config and output directory:

Place your config files as well any other files needed such as datasets or priors in `CONFIG_PATH`. In `OUTPUT_PATH`, the job will write output logs.

```sh
export DATA_PATH=<local_path_to_semlaflow_data_directory>
export OUTPUT_PATH=<local_path_to_save_output>
```

Then, the following command will run the job:

```sh
docker run -it \
  --shm-size=256g \
  --device=/dev/kfd \
  --device=/dev/dri/renderD<RENDER_ID> \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $DATA_PATH:/data \
  -v $OUTPUT_PATH:/output \
  rocm-semlaflow <script> <output-file> \
  --data_path /data/<path-to-dataset> <other_args>
```

## Optimization

Let's talk about the optimizations that we tried in this section. The training times mentioned in this blog are average times per epoch over a total of 200 epochs.

### Docker Image Updates

* We used a ROCm 6.3.3 image initially. This image caused some issues with the multi-processor steps in the semlaflow train and predict scripts. The script would get stuck in the RDKit mols creation stage. Updating the Docker image to `rocm/pytorch:rocm6.4_ubuntu24.04_py3.12_pytorch_release_2.6.0`, solved this issue.
* Updating the image reduced the training time per epoch from `12min34secs` to `10min46secs`.

### Compile

The `torch.compile()` function in PyTorch is designed to enhance model performance by optimizing it through TorchDynamo, which traces model execution and applies various optimizations. When compiling either the entire model or individual modules, the first epoch takes about 15 minutes, but subsequent epochs are significantly faster, balancing out the initial delay.

In the [training script](https://github.com/rssrwn/semla-flow/blob/2f41e571b38c64d44786c04e7a116078b29d3585/semlaflow/train.py#L176), `compile_mode` was initially set to `False`. We enabled compiling by setting `compile_mode` to `True`.

* The original compile command in the [fm.py script](https://github.com/rssrwn/semla-flow/blob/2f41e571b38c64d44786c04e7a116078b29d3585/semlaflow/models/fm.py#L735) is

    ```python
    torch.compile(model, dynamic=False, fullgraph=True, mode="reduce-overhead")
    ```

    Compiling with this initial setup failed due to an error related to CUDAGraphs.

* Instead of compiling the full model, we then tried compiling only the `EquivDynamics` module in `SemlaGenerator` with fullgraph disabled. This worked, but we also encountered the issue of excessive recompilations hitting the cache limit. This can be solved by either increasing the cache limit, or setting dynamic=True.

    This reduced the training time from `10min46secs` to `6min3secs`.
    To compile the `EquivDynamics` module in `SemlaGenerator` module, we added this command in [semla.py](https://github.com/rssrwn/semla-flow/blob/2f41e571b38c64d44786c04e7a116078b29d3585/semlaflow/models/semla.py#L769):

    ```python
    self.dynamics = torch.compile(self.dynamics, dynamic=True)
    ```

* We also tried to compile with mode="max-autotune" which led to a long compilation time, likely due to too many operations to tune and eventually errored out. We didn't pursue this further and similarly didn't use PyTorch's TunableOps for the same reason.

* Later, we determined that the CUDAGraphs error with the initial setup was due to the combination of fullgraph=True and mode="reduce-overhead". The original compile approach does work in the default compile mode, with dynamic=True and fullgraph=True or by setting dynamic=False and increasing the recompilation cache limit. This reduced the training time from `6min3secs` to `5min55secs`

To conclude, compiling the full model with default compile mode, dynamic=True and fullgraph=True gave the best training time.

```python
torch.compile(model, dynamic=True, fullgraph=False)
```

### Compiler Cache Limit

Compiler cache limit is set to 1000. This limit was set based on trial and error, to ensure the cache limit is not reached. There's nothing special about the  value "1000". Change the cache limit from 200 to 1000 in the [Utils script](https://github.com/rssrwn/semla-flow/blob/main/semlaflow/scriptutil.py)

```sh
COMPILER_CACHE_SIZE = 1000
```

Set torch cache size limit in the [Train script](https://github.com/rssrwn/semla-flow/blob/2f41e571b38c64d44786c04e7a116078b29d3585/semlaflow/train.py) based on the compiler cache size value in the utils script.

```sh
torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE
torch._dynamo.config.accumulated_cache_size_limit = util.COMPILER_CACHE_SIZE
```

### No EMA

* Use the `--no_ema` flag to disable the post-batch model averaging callback. This change reduces the overhead introduced by weight averaging at the end of each batch. This reduces the training time per epoch from `5min55secs` to `5min8secs`.

Despite the changes and optimizations applied, the training loss and validation metrics remained fairly consistent.

## Results

| Configuration  | Avg train time / epoch (min)  | Total time (hrs) (Not including validation) | Epochs  | Batch cost
|---|---|---|---|---
|rocm6.3 image|12.57|41.92|200|4096
|rocm6.4 image|10.77|35.92|200|4096
|rocm6.4 image + compiling |5.92|19.75|200|4096
|rocm6.4 image + compiling + no_ema |5.14|17.15|200|4096

## Summary

In this blog, we introduced you to the SemlaFlow package which runs seamlessly out-of-the-box on an AMD MI300X with no code change or engineering effort required. By turning off model averaging after each epoch and compiling the model, we saw a 52% improvement in performance. This blog is the second in the series on running drug discovery use cases on AMD Instinct GPUs. Check out our first blog [here](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html).

For more information about Life Science AI models on AMD hardware, check out the original [article](https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html)

## Additional Resources

The official SemlaFlow repo can be found [here](https://github.com/rssrwn/semla-flow).

The official SemlaFlow paper can be found [here](https://openreview.net/forum?id=bee2G6pEh0).

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
