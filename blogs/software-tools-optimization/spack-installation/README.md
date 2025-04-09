---
blogpost: true
date: 26 Mar 2025
blog_title: "Installing ROCm from source with Spack"
author: 'Garrett Byrd, Joseph Schoonover'
tags: Scientific Computing, HPC, Installation
thumbnail: 'spack-thumbnail.png'
category: Software tools & optimizations
language: English
target_audience: AI/ML and HPC Developers.
key_value_propositions: Users can utilize the Spack package manager to easily install ROCm components from source.
myst:
    html_meta:
        "author": "Garrett Byrd, Joe Schoonover"
        "description lang=en": "Install ROCm and PyTorch from source using Spack. Learn how to optimize builds, manage dependencies, and streamline your GPU software stacks."
        "keywords": "Scientific Computing, HPC, PyTorch"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_developer_type": "ML/AI Developer, Application Developer, HPC Developer"
        "amd_deployment": "Servers"
        "amd_product_type": "Development Tools, Software & Applications"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "High Performance Computing"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Friday Mar 28, 12:00:00 PST 2025
---

# Installing ROCm from source with Spack

In this guide you will learn how Spack makes building ROCm components from source easier and more flexible than other methods. This blog will walk you through installing ROCm from source using the Spack package manager. We will also discuss Spack's place among other [ROCm installation methods](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.2/install/install-overview.html), the landscape of [ROCm components](https://rocm.docs.amd.com/en/docs-6.3.2/what-is-rocm.html), and show you how ROCm, as an open-source software platform, allows developers to streamline software stacks for their applications.

## What is Spack?

From the [Spack website](https://spack.io):
> Spack is a package manager for supercomputers, Linux, and macOS. It makes installing scientific software easy.

Spack began as a from-source package manager, and while the team at Spack are working to provide [binaries for all Spack packages](https://spack.io/spack-binary-packages/), this is still the package manager's greatest strength.

A *from-source* package manager is just that—a package manager that builds its packages from source code (when available). Compare this to a traditional package manager (e.g., `dpkg`/`apt` for Debian-based distros, or `yum`/`dnf` for RHEL-based distros) which installs pre-compiled binaries for each package. Put (extremely) simply, you can think of locally compiled binaries as being "fine-tuned" for the particular machine being used. Because Spack allows you to select the compilers and build optimizations when compiling source code, there is potential to achieve greater performance. Additionally, for GPU accelerated tools and libraries like those included in ROCm, building from source allows you to target specific GPU platforms rather than building for all supported architectures. This can result in reduction in storage costs for installing ROCm, which can be beneficial for creating lightweight container images that depend on ROCm.

## Why install ROCm from source?

If you are installing pre-compiled binaries on a supported operating system (e.g., [using `apt` to install ROCm on Ubuntu 24.04](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.2/install/quick-start.html)), ROCm and its components come pre-built for [any ROCm-supported GPU](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.2/reference/system-requirements.html). These builds are ideal for most Linux package manager installs, so that everyday users need not worry about specific `gfx` versions.

When installed through a traditional package manager, every component is built against every (supported) `gfx` version. For example, [rocBLAS builds against fourteen different `gfx` versions](https://github.com/ROCm/ROCm/blob/29ba151b48c34fa2129a87097936200ab5b494d8/tools/rocm-build/build_rocblas.sh#L34) (including `xnack` variants). For example, if you are developing a ROCm-dependent application for a cluster of AMD Instinct™ MI300X Accelerators, you would only need ROCm components and kernels built against `gfx942`.

There are a few different methods by which you can install ROCm from source, such as [`make`](https://github.com/ROCm/ROCm/blob/develop/tools/rocm-build/README.md)  and [TheRock](https://github.com/ROCm/TheRock), AMD's lightweight open source build system for HIP and ROCm. The advantage of Spack is that it provides incredible ease-of-use. With each release of ROCm, AMD has consistently kept the corresponding Spack packages up to date. Continuing with our MI300X example, we can just `spack install hipblas` to install `hipBLAS`. (We can further influence the specifics of this build process, as we will see below.)

## Install ROCm accelerated packages with Spack

The ROCm Documentation provides a [detailed guide on using Spack](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.2/how-to/spack.html).

Getting started with Spack is quite easy. To install Spack, simply clone the repository and source `setup-env.sh`:

```sh
git clone https://github.com/spack/spack.git ~/spack/
source ~/spack/share/spack/setup-env.sh
```

You must also have C, C++, and Fortran compilers installed. To help Spack find your compilers, you can run the following

```sh
spack compiler find
```

To install `hipBLAS` targeting CDNA3 architectures, you can start by previewing what Spack will install by using `spack spec`

```sh
spack spec hipblas amdgpu_target=gfx942
```

<details><summary>Output of <code>spack spec hipblas amdgpu_target=gfx942</code></summary>

```
 -   hipblas@6.3.2~asan~cuda~ipo+rocm amdgpu_target=gfx942 build_system=cmake build_type=Release generator=make patches=8d71578,b05b34b arch=linux-rocky9-zen4
 -       ^cmake@3.31.6~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release arch=linux-rocky9-zen4
 -           ^curl@8.11.1~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-rocky9-zen4
 -               ^nghttp2@1.65.0 build_system=autotools arch=linux-rocky9-zen4
 -                   ^diffutils@3.10 build_system=autotools arch=linux-rocky9-zen4
 -               ^openssl@3.4.1~docs+shared build_system=generic certs=mozilla arch=linux-rocky9-zen4
 -                   ^ca-certificates-mozilla@2025-02-25 build_system=generic arch=linux-rocky9-zen4
 -           ^ncurses@6.5~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-rocky9-zen4
 -           ^zlib-ng@2.2.3+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-rocky9-zen4
 -       ^compiler-wrapper@1.0 build_system=generic arch=linux-rocky9-zen4
[e]      ^gcc@11.5.0~binutils+bootstrap~graphite~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages='c,c++,fortran' arch=linux-rocky9-zen4
 -       ^gcc-runtime@11.5.0 build_system=generic arch=linux-rocky9-zen4
[e]      ^glibc@2.34 build_system=autotools arch=linux-rocky9-zen4
 -       ^gmake@4.4.1~guile build_system=generic arch=linux-rocky9-zen4
 -       ^hip@6.3.2~asan~cuda~ipo+rocm build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^comgr@6.3.2~asan~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^glx@1.4 build_system=bundle arch=linux-rocky9-zen4
 -               ^mesa@23.3.6+glx+llvm+opengl~opengles+osmesa~strip build_system=meson buildtype=release default_library=shared arch=linux-rocky9-zen4
 -                   ^bison@3.8.2~color build_system=autotools arch=linux-rocky9-zen4
 -                   ^flex@2.6.3+lex~nls build_system=autotools arch=linux-rocky9-zen4
 -                   ^glproto@1.4.17 build_system=autotools arch=linux-rocky9-zen4
 -                   ^libunwind@1.8.1~block_signals~conservative_checks~cxx_exceptions~debug~debug_frame+docs~pic+tests+weak_backtrace~xz~zlib build_system=autotools components=none libs=shared,static arch=linux-rocky9-zen4
 -                   ^libx11@1.8.11 build_system=autotools arch=linux-rocky9-zen4
 -                       ^inputproto@2.3.2 build_system=autotools arch=linux-rocky9-zen4
 -                       ^kbproto@1.0.7 build_system=autotools arch=linux-rocky9-zen4
 -                       ^xextproto@7.3.0 build_system=autotools arch=linux-rocky9-zen4
 -                       ^xproto@7.0.31 build_system=autotools arch=linux-rocky9-zen4
 -                       ^xtrans@1.5.2 build_system=autotools arch=linux-rocky9-zen4
 -                   ^libxcb@1.17.0 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libxau@1.0.12 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libxdmcp@1.1.5 build_system=autotools arch=linux-rocky9-zen4
 -                       ^xcb-proto@1.17.0 build_system=autotools arch=linux-rocky9-zen4
 -                   ^libxext@1.3.6 build_system=autotools arch=linux-rocky9-zen4
 -                   ^libxt@1.3.1 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libice@1.1.2 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libsm@1.2.5 build_system=autotools arch=linux-rocky9-zen4
 -                   ^llvm@17.0.6+clang~cuda~flang+gold~ipo+libomptarget~libomptarget_debug~link_llvm_dylib+lld+lldb+llvm_dylib+lua~mlir+polly~python~split_dwarf~z3~zstd build_system=cmake build_type=Release compiler-rt=runtime generator=ninja libcxx=runtime libunwind=runtime openmp=runtime shlib_symbol_version=none targets=all version_suffix=none arch=linux-rocky9-zen4
 -                       ^binutils@2.43.1~debuginfod~gas+gold~gprofng+headers~interwork+ld~libiberty~lto~nls~pgo+plugins build_system=autotools compress_debug_sections=zlib libs=shared,static arch=linux-rocky9-zen4
 -                       ^hwloc@2.11.1~cairo~cuda~gl~level_zero~libudev+libxml2~nvml~opencl+pci~rocm build_system=autotools libs=shared,static arch=linux-rocky9-zen4
 -                       ^lua@5.3.6+shared build_system=makefile fetcher=curl arch=linux-rocky9-zen4
 -                           ^unzip@6.0 build_system=makefile patches=881d2ed,f6f6236 arch=linux-rocky9-zen4
 -                       ^perl-data-dumper@2.173 build_system=perl arch=linux-rocky9-zen4
 -                       ^swig@4.1.1 build_system=autotools arch=linux-rocky9-zen4
 -                           ^pcre2@10.44~jit+multibyte+pic build_system=autotools arch=linux-rocky9-zen4
 -                   ^py-mako@1.2.4 build_system=python_pip arch=linux-rocky9-zen4
 -                       ^py-markupsafe@2.1.3 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^xrandr@1.5.3 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libxrandr@1.5.4 build_system=autotools arch=linux-rocky9-zen4
 -                           ^renderproto@0.11.1 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libxrender@0.9.12 build_system=autotools arch=linux-rocky9-zen4
 -                       ^randrproto@1.5.0 build_system=autotools arch=linux-rocky9-zen4
 -           ^hipcc@6.3.2~ipo build_system=cmake build_type=Release generator=make patches=c10b010 arch=linux-rocky9-zen4
 -           ^hipify-clang@6.3.2~asan~ipo build_system=cmake build_type=Release generator=make patches=16e0e2b arch=linux-rocky9-zen4
 -           ^libedit@3.1-20240808 build_system=autotools arch=linux-rocky9-zen4
 -           ^numactl@2.0.18 build_system=autotools arch=linux-rocky9-zen4
 -               ^autoconf@2.72 build_system=autotools arch=linux-rocky9-zen4
 -               ^automake@1.16.5 build_system=autotools arch=linux-rocky9-zen4
 -               ^libtool@2.4.7 build_system=autotools arch=linux-rocky9-zen4
 -                   ^findutils@4.10.0 build_system=autotools patches=440b954 arch=linux-rocky9-zen4
 -               ^m4@1.4.19+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rocky9-zen4
 -                   ^libsigsegv@2.14 build_system=autotools arch=linux-rocky9-zen4
 -           ^perl@5.40.0+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rocky9-zen4
 -               ^berkeley-db@18.1.40+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rocky9-zen4
 -               ^bzip2@1.0.8~debug~pic+shared build_system=generic arch=linux-rocky9-zen4
 -               ^gdbm@1.23 build_system=autotools arch=linux-rocky9-zen4
 -           ^perl-file-which@1.27 build_system=perl arch=linux-rocky9-zen4
 -           ^perl-uri-encode@1.1.1 build_system=perl arch=linux-rocky9-zen4
 -               ^perl-module-build@0.4234 build_system=perl arch=linux-rocky9-zen4
 -           ^py-cppheaderparser@2.7.4 build_system=python_pip arch=linux-rocky9-zen4
 -               ^py-ply@3.11 build_system=python_pip arch=linux-rocky9-zen4
 -               ^py-setuptools@76.0.0 build_system=generic arch=linux-rocky9-zen4
 -               ^python-venv@1.0 build_system=generic arch=linux-rocky9-zen4
 -           ^rocm-core@6.3.2~asan~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^rocminfo@6.3.2~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^rocprofiler-register@6.3.2~ipo build_system=cmake build_type=Release generator=make patches=fc2f3cd arch=linux-rocky9-zen4
 -               ^glog@0.7.1~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -                   ^gflags@2.2.2~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^roctracer-dev-api@6.3.2 build_system=generic arch=linux-rocky9-zen4
 -       ^hipblas-common@6.3.2~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -       ^hsa-rocr-dev@6.3.2~asan+image~ipo+shared build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^elfutils@0.192~debuginfod+exeprefix+nls build_system=autotools arch=linux-rocky9-zen4
 -               ^gettext@0.23.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-rocky9-zen4
 -                   ^tar@1.35 build_system=autotools zip=pigz arch=linux-rocky9-zen4
 -                       ^pigz@2.8 build_system=makefile arch=linux-rocky9-zen4
 -               ^libiconv@1.17 build_system=autotools libs=shared,static arch=linux-rocky9-zen4
 -               ^xz@5.6.3~pic build_system=autotools libs=shared,static arch=linux-rocky9-zen4
 -               ^zstd@1.5.6+programs build_system=makefile compression=none libs=shared,static arch=linux-rocky9-zen4
 -           ^libdrm@2.4.124~docs~strip build_system=meson buildtype=release default_library=shared arch=linux-rocky9-zen4
 -               ^libpciaccess@0.17 build_system=autotools arch=linux-rocky9-zen4
 -                   ^util-macros@1.20.1 build_system=autotools arch=linux-rocky9-zen4
 -               ^libpthread-stubs@0.5 build_system=autotools arch=linux-rocky9-zen4
 -               ^meson@1.7.0 build_system=python_pip patches=0f0b1bd arch=linux-rocky9-zen4
 -           ^pkgconf@2.3.0 build_system=autotools arch=linux-rocky9-zen4
 -           ^xxd-standalone@8.2.1201 build_system=makefile arch=linux-rocky9-zen4
 -       ^llvm-amdgpu@6.3.2~ipo~link_llvm_dylib~llvm_dylib+rocm-device-libs build_system=cmake build_type=Release generator=ninja patches=b4774ca arch=linux-rocky9-zen4
 -           ^libxml2@2.13.5~http+pic~python+shared build_system=autotools arch=linux-rocky9-zen4
 -           ^ninja@1.12.1+re2c build_system=generic patches=93f4bb3 arch=linux-rocky9-zen4
 -               ^re2c@3.1 build_system=autotools arch=linux-rocky9-zen4
 -           ^python@3.11.11+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches=13fa8bf,b0615b2,ebdca64,f2fd060 arch=linux-rocky9-zen4
 -               ^expat@2.7.0+libbsd build_system=autotools arch=linux-rocky9-zen4
 -                   ^libbsd@0.12.2 build_system=autotools arch=linux-rocky9-zen4
 -                       ^libmd@1.1.0 build_system=autotools arch=linux-rocky9-zen4
 -               ^libffi@3.4.6 build_system=autotools arch=linux-rocky9-zen4
 -               ^libxcrypt@4.4.38~obsolete_api build_system=autotools arch=linux-rocky9-zen4
 -               ^readline@8.2 build_system=autotools patches=1ea4349,24f587b,3d9885e,5911a5b,622ba38,6c8adf8,758e2ec,79572ee,a177edc,bbf97f1,c7b45ff,e0013d9,e065038 arch=linux-rocky9-zen4
 -               ^sqlite@3.46.0+column_metadata+dynamic_extensions+fts~functions+rtree build_system=autotools arch=linux-rocky9-zen4
 -               ^util-linux-uuid@2.40.4 build_system=autotools arch=linux-rocky9-zen4
 -           ^z3@4.12.4~gmp~ipo~python build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -       ^rocblas@6.3.2~asan~ipo+tensile amdgpu_target=gfx942 build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^hipblaslt@6.3.2~asan~ipo amdgpu_target=auto build_system=cmake build_type=Release generator=make patches=c58195d arch=linux-rocky9-zen4
 -               ^rocm-smi-lib@6.3.2~asan~ipo+shared build_system=cmake build_type=Release generator=make patches=62be726 arch=linux-rocky9-zen4
 -           ^msgpack-c@3.1.1~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^procps@4.0.4+nls build_system=autotools arch=linux-rocky9-zen4
 -           ^py-joblib@1.4.2 build_system=python_pip arch=linux-rocky9-zen4
 -           ^py-msgpack@1.0.3 build_system=python_pip arch=linux-rocky9-zen4
 -           ^py-pip@24.3.1 build_system=generic arch=linux-rocky9-zen4
 -           ^py-pyyaml@6.0.2+libyaml build_system=python_pip arch=linux-rocky9-zen4
 -               ^libyaml@0.2.5 build_system=autotools arch=linux-rocky9-zen4
 -               ^py-cython@3.0.11 build_system=python_pip arch=linux-rocky9-zen4
 -           ^py-virtualenv@20.26.5 build_system=python_pip arch=linux-rocky9-zen4
 -               ^py-distlib@0.3.7 build_system=python_pip arch=linux-rocky9-zen4
 -               ^py-filelock@3.12.4 build_system=python_pip arch=linux-rocky9-zen4
 -               ^py-hatch-vcs@0.4.0 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^py-setuptools-scm@8.2.0+toml build_system=python_pip arch=linux-rocky9-zen4
 -                       ^git@2.48.1+man+nls+perl+subtree~svn~tcltk build_system=autotools arch=linux-rocky9-zen4
 -                           ^libidn2@2.3.7 build_system=autotools arch=linux-rocky9-zen4
 -                               ^libunistring@1.2 build_system=autotools arch=linux-rocky9-zen4
 -                           ^openssh@9.9p1+gssapi build_system=autotools arch=linux-rocky9-zen4
 -                               ^krb5@1.21.3+shared build_system=autotools arch=linux-rocky9-zen4
 -               ^py-hatchling@1.25.0 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^py-editables@0.5 build_system=python_pip arch=linux-rocky9-zen4
 -                       ^py-flit-core@3.10.1 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^py-packaging@24.2 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^py-pathspec@0.11.1 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^py-pluggy@1.5.0 build_system=python_pip arch=linux-rocky9-zen4
 -                   ^py-trove-classifiers@2023.8.7 build_system=python_pip arch=linux-rocky9-zen4
 -                       ^py-calver@2022.6.26 build_system=python_pip arch=linux-rocky9-zen4
 -               ^py-platformdirs@3.10.0~wheel build_system=python_pip arch=linux-rocky9-zen4
 -           ^py-wheel@0.45.1 build_system=generic arch=linux-rocky9-zen4
 -       ^rocm-cmake@6.3.2~ipo build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -       ^rocsolver@6.3.2~asan~ipo+optimal amdgpu_target=gfx942 build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -           ^fmt@11.1.4~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make arch=linux-rocky9-zen4
 -           ^rocsparse@6.3.2~asan~ipo~test amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
 -               ^rocprim@6.3.2~asan~ipo amdgpu_target=auto build_system=cmake build_type=Release generator=make arch=linux-rocky9-zen4
```
</details>

To install, again we just use `spack install hipblas amdgpu_target=gfx942`, this time specifying a `gfx` version.

## Spack environments

A Spack environment is used to group a set of specs intended for some purpose to be built, rebuilt, and deployed in a coherent fashion. You can think of an environment as a blueprint for a set of related software specifications (specs). It defines:

* Which packages to install
* How each package is configured
* Where everything gets installed

Instead of installing packages one-by-one and manually loading modules, environments let you group specs together for a specific purpose; e.g. for creating virtual machine or Docker images for deployment in a [cloud environment](https://www.amd.com/en/solutions/data-center/cloud-computing.html) targeting a specific AMD GPU architecture.

With a single command, you can concretize, install, or activate the entire environment. In concretization, Spack resolves all dependencies and configuration options across your environment. This step prepares everything for installation in a consistent and reproducible way—far more robust than cobbling together ad-hoc scripts. You may have noticed in our previous `spack install hipblas` example above, specifying `amdgpu_target=gfx942` only applied that configuration to the `hipblas` package; this variant does not necessarily propagate to its dependencies. To ensure the `amdgpu_target` option propagates to all packages that have this variant, we can use the `packages.all.prefer: ["amdgpu_target=gfx942"]` option.

Environments also provide stability. Even if upstream Spack packages change, your environment remains unchanged until you choose to re-concretize. That makes it easy to reproduce builds and maintain consistency over time.

By specifying install paths, environments give you a clear filesystem view of what's been installed. Under the hood, Spack still uses a single, deduplicated software installation that can be shared across multiple environments—saving space and reducing rebuilds. Spack Environments can have an associated filesystem view, which is a directory with a more traditional structure, e.g. `<view>/bin`, `<view>/lib`, `<view>/include` in which all files of the installed packages in the environment are linked.

Finally, activating an environment loads only the modules you need for that specific workflow. Spack can even generate shell scripts to automate module loading for the environment—keeping your software environment lean, relevant, and predictable.

As an example, we can set up an environment using a yaml file that defines what packages we want to install, how to install them, and where to install them. Below is an example environment file that installs `hipblas` and all of its dependencies targeting the `gfx942` AMD GPU architecture.

```yaml
spack:
  specs:
  - hipblas@6.3.2
  concretizer:
    unify: true
  packages:
    all:
      prefer:
      - "amdgpu_target=gfx942"
  view: /opt/rocm
```

In this example, hipBLAS and all of its dependencies will be installed in a single Spack managed location and the filesystem view of the packages will be available at `/opt/rocm`. In essence, once this environment is installed, this would provide a hipBLAS installation in much the same way that `apt install hipblas` would.

To work with environments, suppose this example environment file is stored in `$HOME/spack-env/spack.yaml`. You can activate this environment and concretize the build with the following command:

```sh
spack env activate $HOME/spack-env
spack concretize -f
```

To install this environment, simply run:
```sh
spack install
spack gc -y
```
The last step here performs "garbage collection", removing packages that are only build-time dependencies. This reduces image size by removing packages that are not required at runtime and helps lighten up the final installation.

Read more about Spack environments [here](https://spack.readthedocs.io/en/latest/environments.html).

### Using Spack environments to create Docker image recipes

Another major benefit of using a Spack environment is that you can easily containerize it to ensure reproducibility and portability. To do this, you can create a Dockerfile to install just the ROCm packages you care about for the specific architectures you'd like to target. In continuing with our previous example, we can create a Dockerfile by doing the following :

```sh
# Navigate to the directory where your spack.yaml file is located
cd $HOME/spack-env
spack containerize > Dockerfile
```
This creates a two-stage build recipe that starts from Spack managed container images. In the first stage of the build (the "builder" stage), the environment is concretized and installed in the container and build-time dependencies are removed. Following this, all the binary files (executables, shared libraries, and static archives) found under the filesystem view are stripped of their debugging symbols and other non-essential metadata to further reduce the image size. The second build stage builds on a compatible base OS image without Spack installed.  The file system view and installation tree are copied from the "builder stage" and the entrypoint is set to a shell that is preconfigured with the Spack view paths defined in the environment.

The full contents of the Dockerfile generated from this example are shown below.
<details><summary>Output of <code>spack containerize > Dockerfile</code></summary>

```
# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-jammy:develop AS builder

# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir -p /opt/spack-environment && \
set -o noclobber \
&&  (echo spack: \
&&   echo '  specs:' \
&&   echo '  - hipblas@6.3.2' \
&&   echo '  concretizer:' \
&&   echo '    unify: true' \
&&   echo '  packages:' \
&&   echo '    all:' \
&&   echo '      prefer:' \
&&   echo '      - amdgpu_target=gfx942' \
&&   echo '  view: /opt/views/view' \
&&   echo '' \
&&   echo '  config:' \
&&   echo '    install_tree: /opt/software') > /opt/spack-environment/spack.yaml

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment && spack env activate . && spack install --fail-fast && spack gc -y

# Strip all the binaries
RUN find -L /opt/views/view/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . > activate.sh

# Bare OS image to run the installed executables
FROM ubuntu:22.04

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software

# paths.view is a symlink, so copy the parent to avoid dereferencing and duplicating it
COPY --from=builder /opt/views /opt/views

RUN { \
      echo '#!/bin/sh' \
      && echo '.' /opt/spack-environment/activate.sh \
      && echo 'exec "$@"'; \
    } > /entrypoint.sh \
&& chmod a+x /entrypoint.sh \
&& ln -s /opt/views/view /opt/view

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/bin/bash" ]
```
</details>

Learn more about [creating containers with Spack](https://spack.readthedocs.io/en/latest/containers.html).

## Using Spack to understand ROCm dependencies

### Understanding ROCm Components

ROCm is not one piece of software. It is a collection of interconnected, focused components, such the [HIP](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/what_is_hip.html) runtime API for heterogenous systems, [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html) (AMD's BLAS implementation for AMD GPUs), [hipFORT](https://rocm.docs.amd.com/projects/hipfort/en/latest/) (the HIP Fortran library), [various compilers](https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html), and many more application-specific tools and libraries.

ROCm officially supports roughly 70 components, but the exact list of component varies across multiple sources. Here are a few:

- [What is ROCm?](https://rocm.docs.amd.com/en/latest/what-is-rocm.html) $^{[1]}$
- [`ROCm/default.xml`](https://github.com/ROCm/ROCm/blob/develop/default.xml) $^{[2]}$
- [ROCm Documentation for Spack](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/spack.html#rocm-packages-in-spack) $^{[1]}$
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) $^{[1]}$
- [Version release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html#rocm-components) $^{[1]}$

$^{[1]}$ Pages from ROCm Documentation

$^{[2]}$ ROCm GitHub repository

The landscape of ROCm components is best laid out by the following graphic (Figure 1, below), taken from the [What is ROCm?](https://rocm.docs.amd.com/en/latest/what-is-rocm.html) page from the documentation.

<img src="./images/rocm-software-stack-6_3_2.jpg">

<p style="text-align:center">
Figure 1: The ROCm software stack
</p>

The ROCm stack relies on lower layers of runtimes and compilers which are generally essential for most ROCm components; i.e., these are the "meat and potatoes" of ROCm. Above these layers are **Tools**, which are quite handy for developers, but are also used by some ROCm components. The ROCm stack becomes particularly discrete at the **Library** level. E.g., `rocJPEG` does not depend on `hipTensor` and vice versa. This motivates the question, "Why should we install all of ROCm?" if we just need a specific component, and drives us to understand specific dependencies within the ROCm stack.

### ROCm Component Dependencies

As a package manager, Spack is also a great system for understanding the dependencies of the ROCm stack. The following graphic (Figure 2, below) has been generated using the outputs of `spack spec <package>`.

<img src="./images/dep_matrix.png">

<p style="text-align:center">
Figure 2: Dependency matrix for ROCm 6.3.2.
</p>

Light squares indicate that the package on the Y-axis does not depend on the given package on the X-axis. Dark squares indicate that there is a dependency. E.g., `hip` depends on `aqlprofile`, `hipcc`, `hipify-clang`, `hsa-rocr-dev`, `llvm-amdgpu`, `rocm-cmake`, `rocm-core`, `rocminfo`, and `rocprofiler-register`.

Note: Package names are presented as they are in Spack. Exact names may differ from `apt`/`dnf` packages and GitHub repositories.

This graphic covers all of the major ROCm components that are packaged for Spack. We have also included a version of PyTorch + ROCm for reference. The edge here is the verbosity of `spack spec` lets us quickly understand the dependencies of tools that use ROCm, not just the interdependencies of ROCm components.

## Using Spack to install ROCm

Beyond even specific `gfx` versions, your application might not even require all default ROCm components. If you `apt depends rocm6.3.2`, you'll notice that the default installation of ROCm includes `mivisionx`, a set of comprehensive computer vision and machine intelligence libraries, utilities, and applications. Although `mivisionx` comes packaged with ROCm by default, many ROCm components do not require this component. This underscores the utility of selective component installation via Spack.

Unlike `apt` or `dnf`, Spack does NOT maintain a package for a generic `rocm`/`rocm-dev` package. Instead, ROCm components are packaged individually. The following `spack.yaml` approximates an installation of `rocm-dev`. (Note: none of these packages can be built for specific `gfx` versions.)

`spack.yaml`:
```yaml
spack:
  specs:
  - amdsmi
  - comgr
  - hip
  - hipcc
  - hipify-clang
  - aqlprofile
  - hsa-rocr-dev
  - rocm-openmp-extras
  - rocm-cmake
  - rocm-core
  - rocm-dbgapi
  - rocm-debug-agent
  - rocm-device-libs
  - rocm-gdb
  - llvm-amdgpu
  - rocm-opencl
  - rocm-smi-lib
  - rocprofiler-dev
  - rocprofiler-register
  - rocprofiler-sdk
  - roctracer-dev
  concretizer:
    unify: true
  config:
    install_tree: $HOME/opt/rocm
    view: $HOME/opt/rocm
```

It's worth noting that some packages related to ROCm that are available through `dnf`/`apt` are not available in Spack, e.g., `rocm-utils`. Also, some packages such as `aqlprofile` are not open source, and instead these are installed by grabbing the `.deb` from [repo.radeon.com](repo.radeon.com).

### Using Spack to install PyTorch with ROCm

Beyond ROCm, Spack also provides numerous Python packages, including PyTorch. Below is a `spack.yaml` that my be used to build PyTorch and its dependencies specifically for MI300X/MI300A.

`spack.yaml`:
```yaml
spack:
  specs:
  - py-torch+rocm
  concretizer:
    unify: true
  packages:
    all:
      prefer:
      - "amdgpu_target=gfx942"
  config:
    install_tree: $HOME/opt/rocm
    view: $HOME/opt/rocm
```

## Drivers

It is good to remind ourselves that the purpose of the ROCm software stack is to provide developers with the tools to program on (primarily) AMD GPUs. Under the hood, the developer still utilizes drivers to meaningfully interface with their AMD GPU and actually run programs. I.e., Spack and ROCm are used to build applications, drivers are used to run them.

All drivers required for ROCm are fully open source at [`github.com/ROCm/ROCK-Kernel-Driver`](github.com/ROCm/ROCK-Kernel-Driver).

## The Broader ROCm Ecosystem

So far we have described ~70 components (either as `dnf`/`apt`/Spack packages or GitHub repositories) that compose the ROCm software stack. However, if you go to the [ROCm GitHub page](https://github.com/ROCm), you'll find that the ROCm organization has over 300 repositories. Why so many repositories if ROCm only has ~70 components?

This is the beauty of open-source development. AMD is continuously maintaining ROCm integrations, compatibility, backends, etc. for major GPU computing tools, such as PyTorch, JAX, Tensorflow, transformers, Triton, VLLM, just to list a few. In fact **110** of AMD's ROCm repositories are forks (as of publication).

Beyond forks, AMD actively creates new tools for ROCm. Some of these developing tools are  `AITER` (AI Tensor Engine for ROCm), `rocsift` (a C99 debugging API for ROCm), `rocRoller` (a software library for generating AMDGPU kernels), `mxDataGenerator` (a library for data generation indifferent floating point formats), and `TheRock` (a build system for HIP and ROCm).

## Summary

This blog provides a basic introduction to the Spack package manager and how to install ROCm components using Spack. Specific advantages of Spack have been discussed, such as

- Easily building ROCm components from source.
- Installing Spack packages (such as PyTorch) targeting specific `gfx` architectures.
- Using Spack for cleaning up build dependencies.
- Utilizing `spack spec` to understand ROCm dependencies.

This post also highlights the landscape of the ROCm ecosystem and provides an overview of standard and upcoming ROCm components.

## Acknowledgments

Special thanks to [Garrett Byrd](https://github.com/garrettbyrd) and [Dr. Joe Schoonover](https://github.com/fluidnumerics-joe) at [Fluid Numerics](https://www.fluidnumerics.com/) for contributing this blog. The ROCm software ecosystem is strengthened by community projects that enable you to use AMD GPUs in new ways. If you have a project you would like to share here, [please raise an issue or PR](https://github.com/ROCm/rocm-blogs).

### Find Fluid Numerics online at:

- [fluidnumerics.com](www.fluidnumerics.com)
- [YouTube](https://www.youtube.com/@FluidNumerics)
- [GitHub](https://github.com/FluidNumerics)
- [LinkedIn](https://www.linkedin.com/company/fluidnumerics)
- [Reddit](https://www.reddit.com/r/FluidNumerics/)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

