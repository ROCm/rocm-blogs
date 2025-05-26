---
blogpost: true
blog_title: "Introducing ROCm-DS: GPU-Accelerated Data Science for AMD Instinct™ GPUs"
date: 20 May 2025
author: 'Marco Grond, Saad Rahim'
thumbnail: '2025-04-21-ROCm-DS.jpg'
tags: Scientific Computing, Developers, HPC
category: Software tools & optimizations
target_audience: Data Science Users (RAPIDS, Pandas users)
key_value_propositions: This blog serves to inform and attract potential customers of the newly released ROCm-DS toolkit, which enables GPU accelerated data processing operations on Instinct GPUs.
language: English
myst:
    html_meta:
        "author": "Marco Grond, Saad Rahim"
        "description lang=en": "Accelerate data science with ROCm-DS: AMD’s GPU-optimized toolkit for faster data frames and graph analytics using hipDF and hipGRAPH"
        "keywords": "ROCm-DS, RAPIDS, hipDF, cuDF, hipGRAPH, cuGraph, Dataframe, Graph, Data Science, Data Processing, Pandas"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_developer_type": "Data & Research Scientists"
        "amd_deployment": "Servers"
        "amd_product_type": "Software & Applications"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Data Science"
        "amd_industries": "Data Center"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_releasedate": Wed Mar 05, 12:00:00 PST 2025
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

# Introducing ROCm-DS: GPU-Accelerated Data Science for AMD Instinct™ GPUs

AMD is excited to announce the early access release of [ROCm-DS](https://rocm.docs.amd.com/projects/rocm-ds/en/latest/)
(ROCm Data Science), a new toolkit designed to accelerate data processing workloads on AMD Instinct™ GPUs. Built on
the core ROCm toolkit, ROCm-DS promises to significantly enhance performance and scalability for data-intensive
applications, catering to the pressing needs of today’s data-driven landscape. ROCm-DS is based on the
open source libraries in the RAPIDS ecosystem. This collection of libraries
enables a multitude of data processing operations, allowing new and existing workloads to tap into the
computational advantages offered by AMD Instinct Datacenter GPUs. This early access release introduces two
powerful new libraries: [hipDF](https://github.com/rocm-ds/hipDF) and [hipGRAPH](https://github.com/rocm-ds/hipGRAPH).

# GPU Accelerated Data Science

ROCm-DS leverages the computational power of AMD Instinct GPUs to run data science workloads, utilizing all of the advantages
offered by running workloads on GPUs over traditional CPUs. This opens the door to a world of possibilities, allowing
users to run more intensive workloads and process larger datasets in a timely manner. Moreover, existing workloads
can be run on Instinct GPUs with minimal code changes.

This is an ongoing project with additional libraries expected in the latter half of 2025. The ROCm-DS team will continue
to expand and improve this toolkit to offer users even more functionality, enable more data science workloads, and further
optimize projects to provide users with greater performance gains in future releases. With this early access release, users
can experience a small taste of what is to come.

## Why choose ROCm-DS?

ROCm-DS offers users the ability to run existing workloads on AMD Instinct GPUs which in turn enables the expansion of these workloads to
larger datasets. The toolkit provides the following benefits:

- **Accelerated Data Science**: By utilizing high-efficiency GPU processing, ROCm-DS allows for rapid data operations, significantly reducing the time and resources needed for large-scale data processing and analytics.
- **Scalability**: Designed to handle larger datasets and workloads, ROCm-DS offers a scalable solution to meet the growing demands of enterprises and research institutions.
- **Integration and Compatibility**: ROCm-DS seamlessly integrates into existing workflows, supporting a wide array of data processing needs while maintaining compatibility with leading data science libraries and frameworks.
- **Ease of Use**: In addition to accelerating the functionality offered by popular libraries such as [Pandas](https://pandas.pydata.org/), ROCm-DS implements an API compatible with NVIDIA's open source RAPIDS ecosystem, allowing users to effortlessly migrate existing workloads to AMD Instinct GPUs.
- **Established Software**: ROCm-DS is built on top of the larger ROCm ecosystem and utilizes AMD's existing software libraries to deliver proven performance, utilizing leading hardware technology that continues to evolve with the rapid advancements in GPU capabilities.

## GPU accelerated data frames with hipDF

The core of ROCm-DS, [hipDF](https://github.com/rocm-ds/hipDF), enables GPU acceleration for DataFrames and data
manipulation operations. With hipDF, data scientists and engineers can process larger datasets and run more complex
workloads faster than ever, reducing compute time, boosting productivity, and opening up new frontiers for data
processing and analysis. All of this is done with a familiar Pandas-like interface, lowering the barrier to entry
and enabling users to perform fast and efficient data manipulation and analysis.

hipDF provides GPU acceleration for familiar data processing tasks such as filtering, joining, and aggregation,
offering multiple advantages over traditional CPU-based approaches. The API is similar to that of
the familiar [Pandas](https://pandas.pydata.org/) library, and it includes built-in support for accelerating
existing Pandas workloads on GPUs. Furthermore, it implements the same API as the cuDF RAPIDS library, allowing
workloads that have been built on cuDF to be seamlessly transitioned to AMD hardware. For an in depth look into
hipDF, please refer to the [API documentation](https://rocm.docs.amd.com/projects/hipDF/en/latest/reference/hipdf/index.html#hipdf-reference).

Additionally, have a look at our [CuPy and hipDF on AMD: The Basics and Beyond](https://rocm.blogs.amd.com/artificial-intelligence/cupy_hipdf_portfolio_opt/README.html)
or [DataFrame Acceleration: hipDF and hipDF.pandas on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/hipDF_pandas_accelerated/README.html)
blogs for an in depth exploration of some of the functionality enabled by hipDF.

## GPU accelerated graph analytics with hipGRAPH

Taking graph processing to a new level, [hipGRAPH](https://github.com/rocm-ds/hipGRAPH) leverages GPU acceleration to
process and analyze complex graph structures and networks with unparalleled speed and precision. Built on top of hipDF,
hipGRAPH enables users to easily execute a variety of graph processing algorithms on the data stored within the hipDF
dataframes, making it easier than ever to process large-scale graph data efficiently.

hipGRAPH allows users working with large data graphs to access all of the benefits of executing their workloads on AMD Instinct GPUs, enabling them to
process larger and more complex graphs. Whether you are working on social networks, transportation grids, biological
data networks, or any other complex graph networks, hipGRAPH offers a lightning fast solution to perform graph computations
that were previously too computationally complex or time intensive to run on conventional CPUs. hipGRAPH contains a variety
of graph algorithms, including centrality, traversal, similarity, sampling, and labeling algorithms. All of this empowers
users to perform real-time operations and extract deep insights from their data, enabling new applications for these
technologies and assisting decision makers with faster access to results. To get started with hipGRAPH, have a look at
the [API documentation](https://rocm.docs.amd.com/projects/hipGRAPH/en/latest/).

# Using ROCm-DS

Although this release serves as a software preview that has not been optimized for production workloads, we still encourage users to explore
and experience the performance gains that are achievable when accelerating data science workloads with ROCm-DS. In
future releases, new optimizations, functionality, and support will be available to fully utilize the capabilities of
AMD Instinct GPUs to accelerate a wide variety of data science, processing, and analysis workloads.

ROCm-DS offers a Python as well as C++ API, enabling data science users in different fields and professions to
effortlessly utilize the full potential of ROCm-DS. Please refer to either the
[hipDF](https://rocm.docs.amd.com/projects/hipDF/en/latest/) or [hipGRAPH](https://rocm.docs.amd.com/projects/hipGRAPH/en/latest/)
documentation for installation instructions and an exhaustive list of all available functionality included in this early access release.

# Summary

As GPU technology improves, the advantages of leveraging accelerated libraries for data processing workloads will
continue to grow and become a necessity to achieve even greater leaps in data processing capabilities. AMD commits to
enabling Instinct GPUs for the data science industry with ROCm-DS. This early access
release enables users to experience a glimpse of what is to come in the next few years as we continue to expand our
GPU accelerated data science capabilities. Our engineering team is committed to the continued development
and improvement of ROCm-DS, ensuring long lasting support built on an established and robust software and hardware
ecosystem. As ROCm-DS continues to expand and evolve, expect broader library support, enablement of more data science
operations and functions, and greater optimization for production workloads. Additionally, keep an eye on our blogs site
for upcoming blogs for the individual ROCm-DS components. Stay connected for more updates and discover the endless
possibilities that ROCm-DS brings to your data processing solutions.

ROCm-DS joins a family of GPU toolkits supporting multiple industries and verticals. Learn more about our toolkits on
[Instinct docs](https://instinct.docs.amd.com). Stay tuned for upcoming blogs and developments in ROCm toolkits!

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
