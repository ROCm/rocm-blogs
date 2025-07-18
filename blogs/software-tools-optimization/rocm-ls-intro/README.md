---
blogpost: true
blog_title: "Introducing ROCm-LS: Accelerating Life Science Workloads with AMD Instinct™ GPUs"
date: 18 July 2025
author: 'Soumitra Chatterjee, Karthik Kashyap Thatipamula, Deeksha Goplani, Ish Kool, Anik Chaudhuri, Vikas C Sajjan, Marco Grond'
thumbnail: '2025-06-30-rocm-ls.jpg'
tags: AI/ML
category: Software tools & optimizations
target_audience: Customers in the life sciences industry, specifically the medical field
key_value_propositions: This blog serves to inform existing and new life sciences customers of a new vertical specific toolkit applicable to their applications
language: English
myst:
    html_meta:
        "author": "Soumitra Chatterjee, Karthik Kashyap Thatipamula, Deeksha Goplani, Ish Kool, Anik Chaudhuri, Vikas C Sajjan, Marco Grond"
        "description lang=en": "Accelerate life science and medical workloads with ROCm-LS, AMDs GPU-optimized toolkit for faster multidimensional image processing and vision."
        "keywords": "ROCm-LS, hipCIM, cuCIM, Life Sciences, Computer Vision, Image Processing, Multidimensional Images, Medical Imaging"
        "vertical": "Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Data Science"
        "amd_blog_topic_categories": "Industry Applications & Use Cases"
        "amd_blog_authors": "Soumitra Chatterjee, Karthik Kashyap Thatipamula, Deeksha Goplani, Ish Kool, Anik Chaudhuri, Vikas C Sajjan, Marco Grond"
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

# Introducing ROCm-LS: Accelerating Life Science Workloads with AMD Instinct™ GPUs

AMD is thrilled to announce the early access release of [ROCm-LS](https://instinct.docs.amd.com/latest/life-science/index.html)
(ROCm Life Science), a new cutting-edge software toolkit designed to accelerate life science
computational workloads on AMD Instinct™ GPUs. ROCm-LS joins [ROCm-DS](https://instinct.docs.amd.com/latest/data-science/index.html)
as a part of AMD’s new family of toolkits aimed at providing powerful solutions to real world problems.
Similar to ROCm-DS, ROCm-LS is built upon the established ROCm software ecosystem, offering a collection of
components and libraries that address the pressing needs of the life science community. The early access
release of ROCm-LS enables you to experiment with accelerating your life science workloads, such as
digital pathology, automated medical image analysis, and feature extraction and enhancement in large
TIFF files on AMD Instinct GPUs. Join us in exploring this tantalizing glimpse into the future
capabilities of ROCm-LS, setting the stage for the next evolution in life science computing.

This early access release serves as a software technology preview and is not recommended for production
workloads.

# GPU Accelerated Life Science

ROCm-LS utilizes AMD Instinct GPUs to accelerate life science and healthcare workloads. The early
access release serves as the first step in an ongoing effort towards creating a fully fleshed toolkit
which aims to address the needs of the life science and healthcare communities. This release introduces
the [hipCIM](https://github.com/ROCm-LS/hipCIM) library which enhances scientific and medical imaging
applications by leveraging all of the advantages offered by AMD Instinct GPUs.

## What is hipCIM?

hipCIM is an open-source, GPU-accelerated software library which enables computer vision and image
processing operations for multidimensional image datasets. It empowers users from a diverse array of
scientific fields to accelerate their workloads on AMD Instinct GPUs. Some of these fields include
biomedical imaging, geospatial analytics, material sciences, life sciences, remote sensing, and more.

The early access release, based on the open-source cuCIM library, offers an impressive array of
features:

- **GPU-accelerated I/O and Processing Primitives:** hipCIM enhances tasks such as color conversion, feature extraction, filters, morphology, segmentation, and transformations for N-dimensional images.
- **API Compatibility:** hipCIM maintains API compatibility with the NVIDIA cuCIM library, enabling effortless integration and transition of existing workloads to AMD GPUs without the need for "hipification."
- **Multifaceted APIs:** Supporting both Python and C++ APIs, hipCIM facilitates a wide variety of applications and development environments.
- **Mirroring Established Libraries:** The familiar hipCIM API design is similar to the scikit-image and OpenSlide libraries, enabling ease of adoption for users already familiar with these staples.
- **Open Source Collaboration:** Fully open-sourced under the Apache-2.0 license, hipCIM welcomes contributions and fosters a collaborative development environment.

hipCIM provides a powerful solution to accelerate computational vision and image processing
aimed at multidimensional image analysis. It contains a comprehensive set of tools which enable
developers to craft advanced image processing applications by leveraging GPU acceleration. This
not only accelerates important scientific research, but also enables larger datasets to be
processed faster for critical real world applications.

For a better understanding of hipCIM and all it has to offer, be sure to have a look at the
[Announcing hipCIM: A Cutting-Edge Solution for Accelerated Multidimensional Image Processing](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-intro/README.html)
blog.

## ROCm-LS as a Revolutionary Platform

The ROCm Life Science toolkit builds upon the established core ROCm platform, providing a GPU
accelerated alternative to existing life science toolkits. It empowers users to expedite their
life science processing and analysis workloads on AMD GPUs, providing the capability to run
intensive applications involving vast datasets swiftly.

ROCm-LS enables users to:

- **Accelerate New and Existing Workloads:** Boost the performance of life science applications, building scalable solutions tailored to modern, data-centric challenges.
- **Create Advanced Processing Pipelines:** Develop intricate pre- and post-processing applications for AI models and expedite data science workflows.

## Dive Into ROCm-LS: Unlocking Accelerated Computational Vision and Image Processing for Life Sciences

With the early access release of ROCm-LS, AMD invites you to explore and experiment with this
transformative opportunity to delve into enhanced life science workflows. As users engage with
ROCm-LS, they can experience firsthand the remarkable performance improvements that come with
utilizing the full power of AMD Instinct GPUs to accelerate life science workloads. This
release marks the beginning of a promising journey, with future versions set to deliver
additional optimizations, functionalities, and robust support to fully harness the power of
AMD Instinct GPUs across a variety of diverse life science applications.

ROCm-LS features versatile APIs in both Python and C++, catering to professionals and
researchers across various sectors. This dual-language support ensures that users can seamlessly
maximize the toolkit's vast potential, regardless of their field, expertise, or experience. To
explore these capabilities, we recommend consulting the hipCIM documentation, which provides
comprehensive installation instructions and a complete overview of the functionalities available
in this early access release.

Embrace this chance to innovate and accelerate your data-driven life science endeavors with
ROCm-LS, laying the groundwork for more powerful and efficient applications in the future.

## Summary

The early access release of ROCm-LS marks a significant leap forward in the realm of life science
computing on AMD GPUs. This release allows you to accelerate some of your real world life science
workloads such as digital pathology, image preparation for deep learning pipelines, feature
extraction, and tissue segmentation with ROCm-LS and AMDs Instinct GPUs. We invite developers and
researchers to engage with ROCm-LS, explore its capabilities, and contribute to its evolution. Dive
into the future of accelerated scientific computing today with ROCm-LS!

Be sure to check out the latest developments and community contributions hosted on the [ROCm-LS GitHub](https://github.com/rocm-ls).
For more information on contributing or exploring related content, visit the
[ROCm-LS documentation](https://rocm.docs.amd.com/projects/rocm-ls/en/latest/) page,
[Instinct Docs](https://instinct.docs.amd.com/latest/) page, and have a look at the
[Announcing hipCIM: A Cutting-Edge Solution for Accelerated Multidimensional Image Processing](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-intro/README.html)
blog.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
