---
blogpost: true
blog_title: "Getting Started with AMD Resource Manager: Efficient Sharing of Instinct GPUs for R&D Teams and AI Practitioners"
date: 24 Feb 2026
author: 'David Prescott, Akshay Viswanathan, Daniel Gustafsson, Rasmus Larsson'
thumbnail: 'thumbnail_amd_resource_manager.png'
tags: Kubernetes, GenAI, LLM, Serving
category: Software tools & optimizations
target_audience: Platform administrators, Infrastructure administrators, AI practitioners
key_value_propositions: Introduce AMD Resource Manager, its value proposition, how to create and manage projects and workload as well as sharing of compute resources
language: English
myst:
    html_meta:
        "author": "David Prescott, Akshay Viswanathan, Daniel Gustafsson, Rasmus Larsson"
        "description lang=en": "Learn how to utilize the AMD Resource Manager by following this step-by-step guide on how to setup projects, share compute resources and monitor resource utilization."
        "keywords": "Enterprise AI, AMD Resource Manager, AMD Inference Microservices, Kubernetes, Enterprise AI Suite"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "David Prescott, Akshay Viswanathan, Daniel Gustafsson, Rasmus Larsson"
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

# Getting Started with AMD Resource Manager: Efficient Sharing of AMD Instinct™ GPUs for R&D Teams and AI Practitioners

In this blog, you will learn how to use AMD Resource Manager and its components for centralized AI infrastructure governance. It’s part of the AMD Enterprise AI Suite, a full-stack solution for developing, deploying and running AI workloads on a Kubernetes platform designed to support AMD compute. The AMD Resource Manager provides a user-friendly graphical user interface (GUI) and Command Line Interface (CLI) with a unified control plane that simplifies tasks such as managing compute clusters, user access, monitoring resource utilization, and allocating the right compute quotas to the right projects.

This blog covers:

* **AMD Resource Manager**:
  * An introduction to its main components
  * How to set up a project with GPU/compute quotas where you can deploy workloads
  * How to retrieve the `kubeconfig.yaml` file for your cluster
* **Monitor workloads and resource utilization**:
  * How to launch workloads using `kubectl` and monitor them in AMD Resource Manager
* **GPU resource sharing and pre-emption functionality**:
  * An introduction to pre-emption and its benefits for GPU sharing
  * A practical example demonstrating pre-emption in action

## Prerequisites

This blog utilizes the [AMD Enterprise AI Suite](https://enterprise-ai.docs.amd.com/en/latest/index.html). Before proceeding, please ensure the following prerequisites are met:

* **Access to AMD Enterprise AI Suite**: You must have access to an installed instance of the AMD Enterprise AI Suite. Refer to the [Supported Environments]( https://enterprise-ai.docs.amd.com/en/latest/platform-infrastructure/supported-environments.html) documentation for installation details
* [**An overview of the AMD Enterprise AI Suite**](https://enterprise-ai.docs.amd.com/en/latest/platform-overview.html) and [AMD AI Workbench](https://rocm.blogs.amd.com/software-tools-optimization/enterprise-ai-workbench/README.html) is recommended
* **Technical Proficiency**: A working knowledge of command-line or terminal usage and tools such as kubectl and k9s (or similar tools for monitoring the cluster state) is recommended for the latter part of the blog

## AMD Resource Manager

Begin by logging into the **AMD Enterprise AI Suite** as a **Platform Admin** user and navigating to the **AMD Resource Manager** (Figure 1). Before configuring your first project, review the Resource Manager components and their capabilities below:

1. [Dashboard](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/dashboard.html): Provides a high-level overview of your clusters, resource allocations, and basic utilization statistics
2. [Clusters](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/clusters/overview.html): Use to monitor the status and health of your clusters while providing an overview of the resource utilization
3. [Projects](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/projects/manage-projects.html): Create and manage projects, which organize and isolate work within your system. Project settings include user membership, secrets, storage, and quotas
4. [Secrets](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/secrets/overview.html): Manage external secrets for the cluster
5. [Storage](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/storage/overview.html): Manage S3 buckets for the cluster
6. [Users](https://enterprise-ai.docs.amd.com/en/latest/resource-manager/users/overview.html): Manage users, their roles, and their project membership

```{image} ./images/resource_manager_intro.png
:label: resource_manager_overview
:alt: resource_manager _overview
:width: 85%
:align: center
:max-width: 950px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 1: Overview of AMD Resource Manager.</em>
</p>

### Creating a Project with a Quota

We'll begin by setting up and managing a project within AMD Resource Manager. Projects enable you to organize your work on the platform, and each one is kept separate - resources, workloads, and secrets from one project can't be accessed by another. This separation maintains security and keeps each project clearly defined. You can also assign quotas to each project, ensuring users have the necessary resources to run their workloads successfully in shared clusters. A quota defines the ensured amount of resources, such as GPUs, CPUs, and memory, that can be used by workloads within the project. Note that a project can consume additional resources if there are unused resources available.

In this section, we will walk through the process of setting up a new project, configuring basic storage, and adding users.

#### Setting up the Project

First, create a new project with the minimum configuration required to begin deploying workloads.

To create your project:

1. Navigate to the **Projects** page
2. Click on the **Create project** button
3. Enter the project name (“demo-blog-project"), a description, and then select your cluster (see Figure 2 for final details)
4. Click on **Create project**

```{image} ./images/create_project.png
:label: create_project_drawer
:alt: create_project
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 2: The “Create project” dialog box.</em>
</p>

You should receive a confirmation that the project was created successfully and be automatically redirected to the **Project settings**: **Quota** page.

On this page, you can define the quota for the project. As we will adjust these settings in a later section, you may leave the default values unchanged for now.

```{image} ./images/project_created_successfully.png
:label: project_created_successfully
:alt: project_settings_quota
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 3: The Project settings page following successful project creation.</em>
</p>

To ensure storage is configured for your work on the platform, we will now configure access to the default storage in the cluster:

1. Navigate to the **Secrets tab** in the project settings (located next to “Quota”, see Figure 3)
2. Click on the **Add project secret** button and then select **Assign existing secret**
3. Select the **minio-credentials-fetcher** secret from the **Secret** drop-down menu (Figure 4)

```{image} ./images/assign_existing_secret_to_project.png
:label: assign_existing_secret_to_project
:alt: assign_existing_secret_to_project
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 4: The “Assign existing secret” project dialog box.</em>
</p>

4. Click **Assign secret**. You should now see the secret added to the project (Figure 5)

```{image} ./images/project_settings_secrets.png
:label: project_settings_secrets
:alt: project_settings_secrets
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 5: The Project settings “Secrets” page following the addition of the secret.</em>
</p>

Lastly, add existing users to the project:

1. In the project settings, click on the **Users** tab (located next to “Storage”, see Figure 5)
2. Click the **Add Member** button
3. Select yourself (and any other desired users) from the **Users** drop-down menu
4. Click **Add to project**

```{image} ./images/add_users_to_project.png
:label: add_users_to_project
:alt: add_users_to_project
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 6: The "Add User(s) to Project" dialog box.</em>
</p>

The project is now configured, allowing authorized users to deploy workloads within it.

#### Configuring the Project’s Quota

Resource limits (quotas) can optionally be configured for each project to ensure that project members have access to the resources required for their workloads.

```{note}
Projects can consume more than their ensured quota if there are unused resources available. If a workload is submitted to a project that is already consuming its full quota, the system will attempt to borrow resources for it, if available.
When a workload is borrowing resources, it will be suspended if another workload is submitted in a project that has unused quota and there are no other available resources, i.e. the project with unused quota has higher priority for the use of those resources.
```

To configure a project’s quota: If you are still within the project settings, you can simply return to the **Quota** tab. However, to illustrate how to view all available projects and their allocations, the following steps demonstrate an alternative path using the sidebar menu:

1. Navigate to the **Projects** page (see Figure 1). Here you can see all the projects
2. Click on the project (“**demo-blog-project**")
3. Click on **Project Settings** (upper right corner). The **Quota** fields will be displayed, as shown in Figure 3
4. Specify the quotas: **GPUs** = 3, **CPU Cores** = 97, **System Memory** = 800GB, and **Ephemeral Disk** = 150GB. (These quotas will be used later in this blog when demonstrating pre-emption.)
5. Click **Save changes**

#### Obtaining the kubeconfig.yaml for your Cluster

In the next section, we will deploy workloads using `kubectl`. To access the cluster using tools such as kubectl you must obtain the `kubeconfig` file for the cluster. Follow these steps to retrieve the kubeconfig information from the **Clusters** page:

1. Navigate to the **Clusters** page
2. Click on your cluster
3. Click on the **View config** button (upper right corner)
4. Copy the kubeconfig file (Figure 7)
5. Store the file in a secure location on your local machine and set the KUBECONFIG environment variable to point to this file
6. Install and setup the OIDC plugin [kubelogin](https://github.com/int128/kubelogin) using your kubeconfig file

```{image} ./images/cluster_kubeconfig.png
:label: cluster_kubeconfig
:alt: cluster_kubeconfig
:scale: 85%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 7: The “View config” dialog box.</em>
</p>

You should now be able to access your cluster using kubelogin. Let's move on to the next section.

```{note}
Refer to the documentation on [Accessing the Cluster]( https://enterprise-ai.docs.amd.com/en/latest/resource-manager/workloads/accessing-the-cluster.html#logging-in-via-kubectl) for more information.
```

## Monitoring Workloads and Resource Utilization

The AMD Resource Manager allows you to monitor all workloads and resources running within your projects. This includes workloads that you may be managing using a variety of tools such as kubectl, Flyte, Kubeflow, and others. To ensure you receive the full benefits of the AMD Resource Manager - including quota enforcement, access control, and monitoring - workloads are tracked and monitored regardless of how they are submitted to the cluster. This means that workloads submitted via tools, such as kubectl, must adhere to the quotas defined for your project, and you can consistently track and monitor GPU usage and runtime for these workloads.
This tracking and enforcement also apply to Custom Resources that may be created by other operators running within the cluster.
To demonstrate this functionality, we will deploy a workload to the `demo-blog-project` via `kubectl`.

### Prerequisites for Monitoring Workloads and Resource Utilization

* **Project Membership**: Membership to a project. We will be using the “demo-blog-project" created in the sections above
* **Tools**: Ensure [kubectl](https://kubernetes.io/docs/tasks/tools/) and [kubelogin](https://github.com/int128/kubelogin) are installed on your local machine

### Deployment and Monitoring

The deployment process can be seen below:

1. Navigate to the **Project’s Details** page by selecting your project, “demo-blog-project", from the **Projects** page. From this page, we will be able to monitor our soon to be deployed workload (see Figure 8)

```{image} ./images/project_details_no_workloads.png
:label: project_details_no_workloads
:alt: project_details_no_workloads
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 8: The “Project Details” page with no active workloads.</em>
</p>

2. Use the provided Kubernetes manifest below and save it as a file named sample_aims.yaml. While this example utilizes an AMD Inference Microservice (AIM), you may adjust the manifest or replace it with a different workload of your choice. Please note that this manifest will deploy a "meta-llama-llama-3-1-8b-instruct" model

```yaml
apiVersion: aim.silogen.ai/v1alpha1
kind: AIMService
metadata:
  name: sample-aim
spec:
  cacheModel: true
  model:
    ref: amdenterpriseai-aim-meta-llama-llama-3-1-8b-instru-0.8.4-590b84
  replicas: 1
  runtimeConfigName: amd-aim-cluster-runtime-config
```

```{note}
AIMs provide standardized, portable inference microservices for serving AI models on AMD Instinct™ GPUs. They are distributed as Docker images, leverage the ROCm™ software stack and run natively on AMD Instinct™ GPUs, ensuring predictable performance and portability across AMD hardware platforms. Read more in the [AIMs Overview](https://enterprise-ai.docs.amd.com/en/latest/aims/overview.html).
```

3. Submit the workload via kubectl to the namespace matching the name of the project, using the code snippet below:

```bash
kubectl apply -f sample_aims.yaml -n demo-blog-project
```

Return to the project details view; the submitted AIM should now be displayed on the dashboard (see Figure 9)

```{image} ./images/kubectl_workload_dashboard.png
:label: kubectl_workload_dashboard
:alt: kubectl_workload_dashboard
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 9: The “Project Details” page with the automatically tracked workload.</em>
</p>

To view the resource allocation across the entire cluster rather than just a single project, you can monitor workload utilization for all clusters on the **Clusters** page (Figure 10). As you can see, in our case there are now 6 running workloads in total. Please note, that this depends on the actual resource usage, and you may see a different amount of running workloads.

```{image} ./images/clusters_page.png
:label: clusters_page
:alt: clusters_page
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 10: The “Clusters” page showing 6 workloads running in the cluster.</em>
</p>

Clicking on a specific cluster, such as the **demo-cluster**, displays the **Cluster Details** page. This page provides quota and utilization information for the entire cluster and all associated projects (see Figure 11).

```{image} ./images/cluster_details_page.png
:label: cluster_details_page
:alt: cluster_details _page
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 11: The “Cluster Details” page for the “demo-cluster”.</em>
</p>

In addition, by navigating to the **Dashboard** page you can view high-level quota and utilization information for your projects across all clusters, along with live widgets displaying GPU utilization information (see Figure 12).

```{image} ./images/dashboard.png
:label: dashboard_page
:alt: dashboard_page
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 12: The “Dashboard” page showing quota and utilization information.</em>
</p>

Lastly, delete the submitted workload using either kubectl or the AMD Resource Manager UI to clean up the project. To delete the workload via kubectl, use the following command:

```bash
kubectl delete -f sample_aims.yaml -n demo-blog-project
```

Your submitted workload should be deleted. Feel free to go to the Project Details page to confirm.

## GPU Resource Sharing and Pre-emption Functionality

Now that you have set up your project and deployed your first workload, we will demonstrate how the pre-emption or quota functionality works in AMD Resource Manager.

As noted previously, you can allocate a quota for a project by editing the quota in the project settings. Defining the quota ensures a fixed amount of compute resources for the assigned project. You can specify GPU, CPU, system memory, and ephemeral disk allocations for each of your projects.

Resource sharing is handled automatically when a workload is submitted.
Consequently, if project A has an ensured quota, those resources can be **borrowed** by another project, project B, if the ensured quota is not fully utilized by project A.

However, if project A then submits workloads that require the full quota, then project B's workloads that are borrowing from project A get suspended, or pre-empted, and project A’s workloads can be deployed.

If resources subsequently become available on the cluster for project B, then the pre-empted workloads are automatically resumed. For long-running jobs, it is therefore important to have a checkpointing mechanism to avoid losing progress when running on shared resources.

This resource borrowing and pre-emption can be useful in day-to-day work when several teams and types of workloads try to maximize the benefits of limited resources, especially GPUs. For example, by not assigning a quota to CI-jobs, users can submit these jobs at any time, but the jobs will wait in a queue and only run when higher-priority projects are not actively using their allocated resource quota, e.g., during off-peak hours. This approach ensures that computing resources are efficiently utilized while maintaining priority access for high-priority projects.

```{note}
For a workload to pre-empt another workload from a different project, all the resources of the new workload must fit within the quota of the project, this includes GPU, CPU, memory and ephemeral disk (if applicable). If the workload needs to borrow one or more of the resources from other projects to facilitate pre-emption, it will not be scheduled.
```

### Prerequisites for GPU Resource Sharing and Pre-emption Functionality

1. **New project**: Create a new project named **low-prio**. Do not assign any quotas to this project
2. **Quota settings**: Ensure the **demo-blog-project** has the following quota settings:
    * GPUs: 3
    * CPU Cores: 97
    * System Memory: 770 GB
    * Ephemeral Disk: 195 GB

### Practical Illustration of Resource Sharing and Pre-emption

To demonstrate pre-emption and sharing, we will use the following simple workload, which deploys three replicas, each requesting one GPU:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sample-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sample-deployment
  template:
    metadata:
      labels:
        app: sample-deployment
    spec:
      containers:
      - env:
        - name: AIM_CACHE_PATH
          value: /workspace/model-cache
        image: amdenterpriseai/aim-meta-llama-llama-3-1-8b-instruct:0.8.5
        imagePullPolicy: IfNotPresent
        name: inference-container
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        readinessProbe:
          failureThreshold: 3
          periodSeconds: 10
          successThreshold: 1
          tcpSocket:
            port: 8000
          timeoutSeconds: 1
        resources:
          limits:
            amd.com/gpu: "1"
            memory: 48Gi
          requests:
            amd.com/gpu: "1"
            cpu: "4"
            memory: 32Gi

```

Save the Kubernetes manifest above as “sample_deployment.yaml”.
Submit the sample deployment to the newly created **low-prio** project:

```bash
kubectl apply -f sample_deployment.yaml -n low-prio
```

To verify the deployment, open k9s and navigate to the pods view for the **low-prio** namespace (see Figure 13). The deployment is also visible in the AMD Resource Manager where, as shown in Figure 14, three GPUs are currently in use.

```{image} ./images/sample_workload_low_prio_project_running.png
:label: sample_workload_low_prio_project_running
:alt: sample_workload_low_prio_project_running
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 13: k9s - Sample workload running in the low-prio project.</em>
</p>

```{image} ./images/low_prio_running_ui.png
:label: low_prio_running_ui
:alt: low_prio_running_ui
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 14: Workload status as reported in the AMD Resource Manager.</em>
</p>

Now, submit the workload to the original **demo-blog-project**, which has an ensured quota of three GPUs:

```bash
kubectl apply -f sample_deployment.yaml -n demo-blog-project
```

By monitoring the cluster via k9s, we can observe that the workload in the **low-prio** project enters a Pending state (Figure 15). We can also observe the same in the AMD Resource Manager (Figure 16).

```{image} ./images/low_prio_workload_preempted.png
:label: low_prio_workload_preempted
:alt: low_prio_workload_preempted
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 15: k9s - Low-priority workload gets pre-empted.</em>
</p>

```{image} ./images/low_prio_pending_ui.png
:label: low_prio_pending_ui
:alt: low_prio_pending_ui
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 16: The previously running workload is now pending.</em>
</p>

Monitoring the newly deployed workload in **demo-blog-project** confirms that the workload is now running (see Figure 17 for k9s and Figure 18 for AMD Resource Manager). Hence "demo-blog-project" is now using its ensured quota.

```{image} ./images/sample_workload_high_prio_project_running.png
:label: sample_workload_high_prio_project_running
:alt: sample_workload_high_prio_project_running
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 17: k9s - Sample workload submitted to a high priority project.</em>
</p>

```{image} ./images/high_prio_running_ui.png
:label: high_prio_running_ui
:alt: high_prio_running_ui
:width: 85%
:align: center
:max-width: 900px
:class: dark-light
```

<p style="text-align:center">
<em>Figure 18: High priority workload is running. </em>
</p>

You can now clean your environment by either deleting the workload from the AMD Resource Manager or by running the command below:

```bash
kubectl delete -f sample_deployment.yaml -n demo-blog-project
kubectl delete -f sample_deployment.yaml -n low-prio
```

## Summary

In this blog, we covered the basics of project and resource management within AMD Resource Manager. We began by creating and configuring a new project including the quota settings. We then deployed a workload to monitor utilization insights across various dashboard views. Finally, we created a second project to demonstrate the mechanics of resource sharing and pre-emption.

Now that you have configured your first project, try deploying your own custom AI workload or explore the AMD Enterprise AI Suite [documentation](https://enterprise-ai.docs.amd.com/en/latest/) to learn more about its capabilities.

### Get started

AMD Resource Manager is part of the broader AMD Enterprise AI Suite, which provides unified components for scalable inference, resource management, and practitioner tooling. For a greater understanding of the AMD Enterprise AI Suite, see the previous [blog](https://rocm.blogs.amd.com/artificial-intelligence/enterprise-ai-suite/README.html) post.

Ready to build on what you have learned? Use the resources below to start your journey with the AMD Enterprise AI Suite.

For an overview of the AMD Enterprise AI Suite and AIM:

* Visit the [AMD Enterprise AI Suite product page](https://www.amd.com/en/products/software/enterprise-ai-suite.html)
* Visit the [AMD Enterprise AI Suite developer page](https://www.amd.com/en/developer/resources/enterprise-ai-suite.html)
* Visit the [AMD Inference Microservice (AIM): Production Ready Inference on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/artificial-intelligence/enterprise-ai-aims/README.html) blog post
* Visit the [Getting Started with AMD AI Workbench: Deploying and Managing AI Workloads](https://rocm.blogs.amd.com/software-tools-optimization/enterprise-ai-workbench/README.html) blog post

Additional technical documentation from installation to AIM deployment is readily available:

* [Install the AMD Enterprise AI Suite](https://enterprise-ai.docs.amd.com/en/latest/platform-infrastructure/on-premises-installation.html)
* [AMD Resource Manager and AMD AI Workbench](https://enterprise-ai.docs.amd.com/en/latest/platform-overview.html)
* [AIMs catalog](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html)
* [AIM deployment overview](https://enterprise-ai.docs.amd.com/en/latest/aims/deployment_overview.html)
* [AIM custom configuration profiles](https://enterprise-ai.docs.amd.com/en/latest/aims/custom_profiles.html)

If you are interested in fine-tuning, see the following:

* [Low-code fine-tuning in Workbench](https://enterprise-ai.docs.amd.com/en/latest/tutorials/low-code-fine-tuning-tutorial.html)
* [VLM Fine-Tuning for Robotics on AMD Enterprise AI Suite](https://rocm.blogs.amd.com/artificial-intelligence/vlm-finetune-rocm/README.html)
* [Fine-Tune LLMs for Proteins with AMD Enterprise AI Suite](https://rocm.blogs.amd.com/artificial-intelligence/rocm-finetune-protein/README.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

© 2026 Advanced Micro Devices, Inc. All rights reserved.
