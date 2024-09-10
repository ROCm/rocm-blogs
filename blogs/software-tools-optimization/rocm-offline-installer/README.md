---
blogpost: true
date: 10 Sep, 2024
title: 'Introducing the AMD ROCm™ Offline Installer Creator: Simplifying Deployment for AI and HPC'
description: 'Presenting and demonstrating the use of the ROCm Offline Installer Creator, a tool enabling simple deployment of ROCm in disconnected environments in high-security environments and air-gapped networks'
author: Matt Elliott
tags: HPC, Installation
category: Software tools & optimizations
language: English
myst:
  html_meta:
    "description lang=en": "Presenting and demonstrating the use of the ROCm Offline Installer Creator, a tool enabling simple deployment of ROCm in disconnected environments in high-security environments and air-gapped networks."
    "keywords": "HPC, ROCm, MI300, libraries, installation"
    "property=og:locale": "en_US"
---

## Introducing the AMD ROCm™ Offline Installer Creator: Simplifying Deployment for AI and HPC

<span style="font-size:0.7em;">10 Sep, 2024 by {hoverxref}`Matt Elliott<mattelli>` </span>

In the fast-paced world of deep learning and high-performance computing, efficient deployment of drivers and software is crucial. AMD's ROCm™ platform has become a cornerstone for developers and researchers leveraging the power of AMD Instinct Accelerators. However, deploying ROCm at scale in air-gapped or high-security environments without internet access has been a challenge – until now.

AMD presents the ROCm Offline Installer Creator, a practical tool designed to simplify the deployment of ROCm in disconnected environments. This solution addresses a common challenge, offering a straightforward approach to creating customized offline installation packages. By providing this tool, AMD is taking another step towards enabling AI at scale, ensuring efficient ROCm deployment across various environments, including those with limited or no internet access. This blog post will show you how to deploy the ROCm Offline Installer Creator to generate and use an installation package for ROCm and the AMDGPU linux kernel driver in an offline environment.

### The need for offline installation

The ROCm Offline Installer Creator addresses a long-standing challenge in managing large compute environments, offering a streamlined solution for offline deployment of the ROCm software stack. High-security environments often restrict external network connections, while remote or isolated facilities might have limited internet access. Air-gapped systems in sectors such as defense and finance also require solutions that can be deployed without an internet connection. The ROCm™ Offline Installer Creator efficiently addresses these needs. By enabling offline installation, AMD solves a practical problem and enhances the overall user experience with a more streamlined and error-resistant process.

### Key features

The ROCm Offline Installer Creator brings several useful features to the table:

* **Customization:** Tailor the installer to your specific deployment needs.  
* **Simple UI:** An intuitive menu-driven system for easy setup.  
* **Multi-Distro support:** Compatible with Ubuntu (20.04, 22.04, 24.04), RedHat Enterprise Linux 8 and 9, and SUSE Linux Enterprise Server 15 SP5 and SP6.  
* **ROCm version selection:** Choose specific ROCm versions and components to install.  
* **AMDGPU driver integration:** Install and configure the AMDGPU driver and set preferences for driver behavior after installation.  
* **Efficient dependency management:** Automatically resolve and package dependencies into the offline installer package.

### Getting started

Creating and using an offline installer package is straightforward:

* [**Review the documentation**](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/rocm-offline-installer.html) to familiarize yourself with the options available in the tool.  
* **Download the ROCm Offline Installer Creator** from [https://repo.radeon.com/rocm/installer/rocm-linux-install-offline/](https://repo.radeon.com/rocm/installer/rocm-linux-install-offline/). Alternatively, [follow the documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/install/rocm-offline-installer.html\#building) to download and compile the code from [https://github.com/ROCm/rocm-install-on-linux/](https://github.com/ROCm/rocm-install-on-linux/).  
* **Run the ROCm Offline Installer Creator** on a compatible system with internet access and use the interface to select the desired ROCm version, components, and options.  
* **Create a self-contained installer package** and run it on target systems.

### Example usage

Here is the tool in action on an Ubuntu 22.04 server. As detailed in the [instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/rocm-offline-installer.html\#getting-started), download and run the pre-compiled binary to display the Offline Installer Creator UI.

<div id="offline-tool-1"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/offline-tool-1.cast', document.getElementById('offline-tool-1'));
</script>

Customize the Offline Installer package by navigating through the sections in the UI. Most options can be toggled with the **Enter** key, and the **Space Bar** selects multiple options. After the desired settings are chosen, choose `< CREATE OFFLINE INSTALLER >` to review the selections. `<ACCEPT>` begins the process of downloading components and building the customized Offline Installer package.

<div id="offline-tool-2"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/offline-tool-2.cast', document.getElementById('offline-tool-2'));
</script>

The Offline Installer Creator downloads all necessary software and dependencies, bundling them into a file named `rocm-offline-install.run` by default.

<div id="offline-tool-3"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/offline-tool-3.cast', document.getElementById('offline-tool-3'));
</script>

Copy `rocm-offline-install.run` to the desired systems and run it. ROCm software, accelerator drivers, and related tools are installed on the target system without the need to download files from an external source.

<div id="offline-install"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/offline-install.cast', document.getElementById('offline-install'));
</script>

### Conclusion

In this blog post we demonstrated, step-by-step, how to use and deploy the ROCm Offline Installer Creator. This tool represents a step forward in AMD's commitment to streamlining the deployment of infrastructure, enabling HPC and AI development to be more accessible and efficient. AMD invites you to try the ROCm Offline Installer Creator and experience the benefits firsthand. Learn more at the [ROCm Offline Installer documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/rocm-offline-installer.html) page.
