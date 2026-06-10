---
blogpost: true
blog_title: "Streamlining Recommendation Model Training on AMD Instinct™ GPUs"
date: 2 Mar 2026
author: 'Tharun Adithya Srikrishnan, HongTao Meng, Yue Liu, Claire Lee, Gene Su, Ziqiong Liu, Yao Fu, Steve Reinhardt'
thumbnail: 'recsys_thumbnail.jpg'
tags: Recommendation Systems, AI/ML
category: Applications & models
target_audience: Engineers within companies training recommendation models
key_value_propositions: Simplifying recommendation training by packaging all requirements into a docker image, Highlighting AMD advantages for recommendation models
language: English
myst:
    html_meta:
        "author": "Tharun Adithya Srikrishnan, HongTao Meng, Yue Liu, Claire Lee, Gene Su, Ziqiong Liu, Yao Fu, Steve Reinhardt"
        "description lang=en": "Explore how the ROCm training docker can be used for recommendation model training on Instinct GPUs, along with a guide on configuring the workload."
        "keywords": "Recommendation Systems, Distributed Training, Training Docker"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Tharun Adithya Srikrishnan, HongTao Meng, Yue Liu, Claire Lee, Gene Su, Ziqiong Liu, Yao Fu, Steve Reinhardt"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

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

# Streamlining Recommendation Model Training on AMD Instinct™ GPUs

Recommendation model training and inference workloads represent a
significant portion of computational requirements across industries
including e-commerce, social media and content streaming platforms.
Unlike LLMs, recommendation models result in to complex and often imbalanced
communication across GPUs, along with a higher load on the CPU-GPU
interconnect. The [ROCm training
docker](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html?model=pyt_train_dlrm) [[1]](#ref1)
now includes essential libraries for recommendation model training. This
blog demonstrates the functionality and ease of training recommendation
models using ROCm, along with suggestions for improved configuration of
these workloads. We also highlight the inherent benefits of the large
HBM size on AMD Instinct™ GPUs for recommendation workloads.

## Deep Learning Recommendation Model (DLRM)

DLRMv2 [[2]](#ref2) is a representative model for recommendation systems that handles
dense (numerical) and sparse (categorical) features along separate
paths, then combines them for click-through rate prediction. The
architecture includes a bottom MLP for dense features, embedding tables
that transform sparse categorical features into dense representations,
and a top MLP for joint processing.

## Sparse Embeddings

Embeddings are stored in large embedding tables that may reside entirely
on a single GPU, sharded across multiple GPUs, or partially offloaded to
CPUs. Processing a single input can require fetching table entries
spread across different GPUs, which creates complex and imbalanced
communication patterns. TorchRec provides primitives to handle these
distributed embeddings, and FBGEMM accelerates the lookup operations.
The ROCm training docker comes pre-installed with libraries needed for
high-performance computation, communication and sparse embedding
operations for recommendation workloads.

## Configuration of Table Sharding

Selecting the appropriate sharding scheme for a given set of tables is key to
improving performance. We summarize some of the sharding schemes
implemented in TorchRec, as shown in figure 1 below. In the simplest data parallel (DP)
scheme, each rank holds a complete copy of the table. Table-wise (TW)
places a table on a single rank. Row-wise (RW) is preferred for tables
with a larger number of rows, while Column-wise (CW) sharding is preferred
when the embedding dimension is large. Each of these sharding schemes
offers a trade-off between communication complexity, load imbalance and
memory. The sharding planner in TorchRec uses a performance model based
on the system configuration (interconnect bandwidths, memory) to
distributetables across ranks for optimized end-to-end training and
inference performance.

```{figure}  ./images/image1.png
:align: center
:alt: Table Sharding Schemes
:width: 75%

Figure 1: Embedding Table Sharding Schemes [[3]](#ref3)

```

With communication load being a bottleneck on recommendation workloads,
the large HBM on AMD GPUs allows for placing a higher fraction of the
tables locally via DP. It is thus essential to configure the system
specifications for the planner to select the optimal scheme through the
Topology class. These impacts become more pronounced when running training or inference
on larger multi-node setups. For example, for a single node consisting of 8x MI300 GPUs:

```python
# Configuring the Topology class for a single node with 8x MI300 GPUs:

hbm_cap = 192 * 1024 * 1024 * 1024  # 192GB MI300X memory size
ddr_cap = 1024 * 1024 * 1024 * 1024  # Up to 1TB host memory per GPU (System Specific)
hbm_mem_bw = 5.3 * 1024 * 1024 * 1024 * 1024 / 1000  # 5.3 TB/s MI300X
ddr_mem_bw = 0.8 * 460.8 * 1024 * 1024 * 1024 / 1000  # ~370 GB/s (System Specific: For e.g., 80% of the theoretical 460.8 GB/s of AMD EPYC™ 9654)
hbm_to_ddr_mem_bw = 128 * 1024 * 1024 * 1024 / 1000  # 128 GB/s (PCIe gen5x16)
intra_host_bw = 0.8 * 7 * 64 * 1024 * 1024 * 1024 / 1000  # ~336 GB/s (80% of 7x64 GB/s using AMD's xGMI)

topology = Topology(
    local_world_size=get_local_size(),
    world_size=dist.get_world_size(),
    compute_device=device.type,
    hbm_cap=hbm_cap,
    ddr_cap=ddr_cap,
    hbm_mem_bw=hbm_mem_bw,
    ddr_mem_bw=ddr_mem_bw,
    hbm_to_ddr_mem_bw=hbm_to_ddr_mem_bw,
    intra_host_bw=intra_host_bw,
)

```

## DLRM Training Using ROCm Training Docker

We now provide a quick demonstration of the ease of training recommendation models on ROCm.
We use the [ROCm training
docker](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html?model=pyt_train_dlrm)
which comes pre-installed with the required libraries. We have provided
an example for single node DLRM training using the DLRM_v2 model at
<https://github.com/AMD-AGI/DLRMBenchmark>.

1. Clone the repository:

   ```bash
   git clone https://github.com/AMD-AGI/DLRMBenchmark.git
   ```

2. Pull the ROCm training docker container:

   ```bash
   docker pull rocm/primus:v26.1

   ```

3. Launch the container. Ensure all required paths, including the codebase, are mounted (similar to /home_dir/).

   ```bash
   docker run -d \
   --ipc=host \
   -v /dev/shm:/dev/shm \
   -v /home_dir/:/home_dir/ \
   -e USER=$user -e UID=$uid -e GID=$gid \
   --device=/dev/kfd \
   --device=/dev/dri \
   --device=/dev/infiniband \
   --ulimit memlock=-1:-1 \
   --shm-size 32G \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --group-add video \
   --network=host \
   --name dlrm_demo \
   -it rocm/primus:v26.1 \
   tail -f /dev/null
   ```

4. Start interactive shell session within container:

   ```bash
   docker exec -it dlrm_demo bash

   ```

5. Launch training via the single node training script in the repository. Note that a training configuration is available at ./training_config.sh.

   ```bash
   ./launch_training_single_node.sh

   ```

During a successful run, the training log shows stable performance:

```text
Epoch 0:   8%|▊         | 75/1000 [02:01<00:42, 21.78it/s]
 Mean loss: 0.69320858
 Mean loss: 0.69374704
 Mean loss: 0.69327664

Epoch 0:   8%|▊         | 78/1000 [02:01<00:41, 22.03it/s]
 Mean loss: 0.69344836
 Mean loss: 0.69318008
 Mean loss: 0.69339764

```

The train_config.sh file can be updated to point to the Criteo-1B data if available. The training loss then converges, as shown in figure 2 below:

```{figure} ./images/image2.png
:align: center
:alt: DLRM Training Convergence
:width: 67%

Figure 2: DLRM Training Convergence

```

## Summary

Recommendation model training on ROCm has been simplified through the
training docker, pairing TorchRec and FBGEMM with high-performance
communication and computation kernels. Properly configuring the TorchRec
sharding planner allows the system to exploit AMD Instinct™ GPU's large
HBM to favor local placement and reduce communication bottlenecks. We
provided an example of training recommendation models through the DLRM
workload. These capabilities extend to larger multi-node deployments,
where system-aware sharding can sustain performance at scale. With this,
we hope that teams can accelerate recommendation model development and
achieve strong end-to-end performance on AMD GPUs and ROCm!

## References

<a id="ref1"></a>[1] [Training a model with
PyTorch on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html?model=pyt_train_dlrm)

<a id="ref2"></a>[2] Maxim Naumov and Dheevatsa Mudigere and Hao-Jun Michael Shi and
Jianyu Huang and Narayanan Sundaraman and Jongsoo Park and Xiaodong Wang
and Udit Gupta and Carole-Jean Wu and Alisson G. Azzolini and Dmytro
Dzhulgakov and Andrey Mallevich and Ilia Cherniavskii and Yinghai Lu and
Raghuraman Krishnamoorthi and Ansha Yu and Volodymyr Kondratenko and
Stephanie Pereira and Xianjie Chen and Wenlin Chen and Vijay Rao and
Bill Jia and Liang Xiong and Misha Smelyanskiy (2019). Deep Learning
Recommendation Model for Personalization and Recommendation Systems.
<https://arxiv.org/abs/1906.00091>

<a id="ref3"></a>[3] Mudigere, Dheevatsa and Hao, Yuchen and Huang, Jianyu and Jia,
Zhihao and Tulloch, Andrew and Sridharan, Srinivas and Liu, Xing and
Ozdal, Mustafa and Nie, Jade and Park, Jongsoo and Luo, Liang and Yang,
Jie (Amy) and Gao, Leon and Ivchenko, Dmytro and Basant, Aarti and Hu,
Yuxi and Yang, Jiyan and Ardestani, Ehsan K. and Wang, Xiaodong and
Komuravelli, Rakesh and Chu, Ching-Hsiang and Yilmaz, Serhat and Li,
Huayu and Qian, Jiyuan and Feng, Zhuobo and Ma, Yinbin and Yang, Junjie
and Wen, Ellie and Li, Hong and Yang, Lin and Sun, Chonglin and Zhao,
Whitney and Melts, Dimitry and Dhulipala, Krishna and Kishore, KR and
Graf, Tyler and Eisenman, Assaf and Matam, Kiran Kumar and Gangidi, Adi
and Chen, Guoqiang Jerry and Krishnan, Manoj and Nayak, Avinash and
Nair, Krishnakumar and Muthiah, Bharath and khorashadi, Mahmoud and
Bhattacharya, Pallab and Lapukhov, Petr and Naumov, Maxim and Mathews,
Ajit and Qiao, Lin and Smelyanskiy, Mikhail and Jia, Bill and Rao, Vijay
(2022). Software-hardware co-design for fast and scalable training of
deep learning recommendation models. ISCA '22: Proceedings of the 49th
Annual International Symposium on Computer Architecture. Pages 993 --
1011.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
