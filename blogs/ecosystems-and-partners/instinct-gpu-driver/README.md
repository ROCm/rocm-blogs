---
blogpost: true
blog_title: "ROCm Gets Modular: Meet the Instinct Datacenter GPU Driver"
date: 11 Apr 2025
author: 'Saad Rahim, Danny Guan'
thumbnail: 'divergence.png'
tags: Installation
category: Ecosystems and Partners
target_audience: All Instinct GPU customers are the target audience.
key_value_propositions: A fundamental shift in our software release strategy to allow users to distinguish the Instinct GPU driver from the ROCm userspace. New applications where users do not need the ROCm userspace such as K8s, ISV applications and virtualization benefit from a separate driver distribution. Existing users will see the relationship between a single Instinct driver and multiple ROCm userspace versions clearly. In addition, future open source users can clearly differentiate upstream AMDGPU drivers from AMD's Instinct driver distribution.
language: English
myst:
    html_meta:
        "author": "Saad Rahim, Danny Guan"
        "description lang=en": "We introduce the new Instinct driver-a modular GPU driver with independent releases simplifying workflows, system setup, and enhancing compatibility across toolkit versions."
        "keywords": "Instinct, GPU, amdgpu, ROCm, toolkit"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_applications": "Deploying AI at Scale"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_developer_type": "Software Developer"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_product_type": "Accelerators"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_applications": "Cloud Computing"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Tues Mar 18, 12:00:00 PST 2025
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

# ROCm Gets Modular: Meet the Instinct Datacenter GPU Driver

Today ROCm is synonymous with software for AMD's Instinct GPUs. ROCm describes
everything from the driver to the runtime to the libraries that enable AI and HPC software stacks.
Starting in ROCm 6.4, we expand our software family to include the Instinct Datacenter GPU driver.
The Instinct driver bifurcates from the current ROCm driver with a separate release process including an independent
version number scheme, a new documentation site, and a laser focus on enabling applications on our datacenter GPU products. This change is depicted in the figure below.

```{figure} images/support-flow.jpg
:alt: Driver and ROCm Bifurcation
:align: center

Figure 1: Instinct Driver and ROCm Toolkit Bifurcation
```

This blog will help you prepare for these changes and mitigate impacts. You will learn how the new [Instinct Driver](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/) makes your software deployment, container usage, and long-term maintenance more flexible and future-ready.

## What is the Instinct driver?

AMD's open source GPU driver is available through several channels. Users currently get the driver, amdgpu, through
ROCm releases, through Radeon Software for Linux and it is present in many Linux kernel builds. The build of
the amdgpu driver and related packages currently distributed and documented with ROCm, is now renamed as
the Instinct driver. Previously, it was referred to as the amdgpu, ROCm driver or ROCk. The source code of the driver
is published on [ROCm/ROCK-Kernel-Driver](https://github.com/ROCm/ROCK-Kernel-Driver) (soon to be renamed to ROCm/instinct-driver). You may ask, "This is just
a renaming, why do I care?". The changes that happen with ROCm 6.4 are nomenclature related and may be ignored
without repercussion for now. In the future, the Instinct driver will focus exclusively on the subset of features needed
for headless datacenter GPUs (also referred to as accelerators or AI cards), i.e. GPUs without a display out. New and exciting futures are planned for the Instinct driver:

- New installation options to remove permission complexities such as user membership in the video or render groups.
- Future installation options may exclude packages needed to run display outputs to reduce the driver footprint.
- A future driver release series may be maintained for security fixes for an extended period as long term stability driver.
- AMD-SMI and other system management components currently included in ROCm will transition to the Instinct driver releases in the future.
- Users choosing to use amdgpu from the stock Linux kernels may choose to skip all the installation documentation for ROCm that references the Instinct driver. Please note this is not an AMD support option today.

Please note that the Instinct drivers will not take steps to exclude products from other AMD GPU families although certain features may be limited by hardware capabilities.

## Why separate?

Software modularity is key to improving usability for our users. Splitting the driver makes it clear
that a single version of the driver can support software development with multiple versions of ROCm toolkits. We will refer to the current ROCm userspace components as the ROCm toolkit. This also
means you can run software built against multiple versions of ROCm toolkits without upgrading or downgrading the driver based on our Instinct driver support policy.

The Instinct driver support policy establishes forward and backward requirements compatibility between the
Instinct driver with the ROCm toolkit. From a driver release perspective, compatibility is maintained for ROCm toolkit
releases up to one year prior to the driver release and ROCm toolkits released up to one year after the driver release. From the
ROCm toolkit perspective, the same applies. A ROCm toolkit continues support with an Instinct driver release one year after the toolkit's release.

Separate releases for the Instinct driver and ROCm toolkits allows us to fix and release  Instinct driver issues independently
from ROCm release timelines. Today, due to our unified release structure, fixes in the driver are not released independent of a single ROCm release.

### Software Life Cycle Policy

To better understand the new modular approach of the ROCm toolkit and the Instinct driver, let's take a look
at the new Software Life Cycle in Figure 2.

```{figure} images/support-matrix.jpg
:label: supportmatrix
:alt: Software Support Matrix
:align: center

Figure 2: Instinct Driver and ROCm toolkit support life matrix. The vertical axis shows the different ROCm toolkit release versions, the horizontal axis shows the Instinct driver release versions. The label “GA” on the graph represents the general availability release. The colored bars illustrate the computability window for each ROCm toolkit version and its corresponding Instinct driver version, their length represents the duration of the software support window.
```

```{note}
Future ROCm toolkit and Instinct driver release timelines are not finalized. This figure illustrates 
the support policy and compatibility but does not confirm the exact future release schedule or naming conventions, which are subject to change.
```

### Support Policy Changes

The new software support window will offer significant advantages for software modularity and usability for
all users. Having a single Instinct driver capable of supporting multiple ROCm toolkit versions will
greatly simplify the overall system software management, especially in environments where developers
may be working with multiple ROCm toolkit versions.

This Instinct driver support duration change will reduce the need for frequent driver updates when
switching development between different ROCm toolkit versions. As the graph depicts, the ROCm toolkit duration
on backward and forward compatibility support window will be increased starting ROCm 6.4, though Instinct Driver from ROCm 6.0 will not support ROCm toolkit 6.4.

Compared to the previous software support policies, where the ROCm driver support would typically range between
two future and past ROCm toolkit releases (total of 6 months backwards/forwards), this new software support policy will now encompass
a total of 4 ROCm toolkit releases (total of 12 months backwards/forwards). You can observe this change through the
diagram where earlier ROCm toolkit releases like 6.0 and 6.1 have shorter compatibility windows, reflecting
the older policy, but transitions to a much longer support window, starting from ROCm 6.4, show casing the expanded
one-year support window.

## The transition

Separation of the software releases occurs in phases. During the ROCm 6.4 release, we create a separate
documentation site for the Instinct driver while keeping the version number for the Instinct driver synced
with ROCm. Coinciding with the ROCm 6.5 release later this year, the version numbering of the Instinct
driver release diverges from ROCm.

```{note}
The transition plan does not change or introduce any new features, it is a change designed to make our
capabilities more visible.
```

## Instinct driver version scheme

Package names continue to include the component version numbers in their
naming schemes, maintaining the current structure. We repurpose the ROCm
upgrade version field into a 8-digit field to represent the common AMDGPU
version number (shared between Radeon and Instinct releases), taking into
account the point release numbering structure.
For instances, a Instinct driver release might be versioned as 30.10.2.
This results in the following package versions:

> Package Naming:
>
> - Kernel Driver: amdgpu-dkms_6.10.5.30100200-2109964.24.04_all.deb
> - UMD Component (with it's own version number): libgl1-amdgpu-mesa-glx_24.3.0.30100200-2109964.24.04_i386.deb
> - Meta-Packages (Patch and Point release numbers included): amdgpu-core_30.10.2.30100200-2109964.24.04_all.deb

The "Instinct Driver" will be referred to as "Instinct Driver 30.10.2".
The path in repo.radeon.com will use the new version number, for example https://repo.radeon.com/amdgpu/30.10.2/.

The reason for the change to this new version numbering scheme is to avoid confusion between the Instinct driver and ROCm toolkit, thus
they both need to be versioned completely differently. We are following [semver](https://semver.org/)
for the version numbering, using Major.Minor.Patch as the version number system. For the first Instinct Driver release,
the semantic version Major number begins with 30, incrementing by 1 every year, while the Minor number
starts at 10, and will increment by 10 each Minor release (approximately once a quarter).

## Common installation scenarios

Let's go over a few installation scenarios made clearer through this nomenclature change. Please note that the all-in-one installation option for ROCm with the Instinct driver is not going away.

### 3rd party applications or containers

Many of our [software ecosystem application partners](https://instinct.docs.amd.com/latest/isv-apps/index.html) bundle ROCm binaries in their installers.
Installing ROCm is redundant in this situation. Currently, the same version number is used between the ROCm userspace and ROCm driver due to the linked release.
Many users end up mistakenly installing all of ROCm to enable their application. With easier to follow instructions, the user installs the Instinct driver and
runs the ISV installer.

Our nomenclature changes also make it clear to container users that you don't need to install ROCm to run a container. Just the Instinct
driver suffices.

### Managing a server with multiple ROCm versions

A server with multiple versions of the ROCm toolkit and one Instinct driver version is often used by developers.
Let's imagine a system with 3 versions of ROCm installed: ROCm 6.1.0, ROCm 6.2.0, and ROCm 6.4.0. The user also installs
Instinct driver 30.10.0 in the summer of 2025. This newly installed Instinct driver will support all ROCm versions released one year earlier,
and one year later. In the Fall of 2025, the user may install ROCm 7.0 and will not need to upgrade the driver. Alternatively, the user may
upgrade the driver to Instinct 30.20.0 and still have the currently installed ROCm versions fully functional.

### Build server

Scenarios where a server just builds code can install the ROCm toolkit. As ROCm no longer means installing the driver,
users may make the choice to skip the driver installation easily.

## Summary

As we showed, starting with the ROCm 6.4 release you can expect a smoother AI and HPC development experience, as AMD is making software for Instinct GPUs more modular by separating
the user space components from the driver space components. Previously, ROCm releases and its associated
versioning described both the userspace and driver. In ROCm 6.4, the documentation for the driver space
and system management moves to instinct.docs.amd.com. Information on the variant of the amdgpu driver built
for Instinct GPUs is available on the Instinct driver website. Source code is available at ROCm/ROCK-Kernel-Driver
(with an upcoming repo rename to instinct-driver). For the 6.4 release, the versioning scheme for the Instinct driver
does not change. It is referred to as the Instinct driver version 6.4. In the subsequent ROCm minor release,
currently planned as ROCm 6.5, the Instinct driver version will separate from ROCm versioning.

Our journey to make the ROCm ecosystem modular is not stopping with the Instinct driver. We anticipate further reorganization of existing products in the ROCm ecosystem, such as separate toolkits targeting with HPC markets and computer vision. New toolkits will arrive that open up new capabilities in storage, data science and life science. We are just starting!

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
