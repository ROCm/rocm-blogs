cd /workspace

# Install dependencies
apt-get update && apt-get install -y git-lfs cmake
git lfs install

pip install --upgrade pip setuptools wheel scikit-build-core build setuptools-scm
pip install "torch_geometric>=2.5" nanobind phonopy --no-build-isolation

# Install MatterGen
git clone https://github.com/microsoft/mattergen.git
cp /workspace/src/mattergen-pyproject.toml /workspace/mattergen/pyproject.toml
cp /workspace/src/config_updates.patch /workspace/mattergen/config_updates.patch
cd mattergen
git apply config_updates.patch
pip install -e . --no-build-isolation

# Install Torch Scatter; MI300X corresponding architecture is gfx942
cd /workspace
git clone https://github.com/silogen/pytorch_scatter.git
cd pytorch_scatter
git checkout remotes/origin/feature/hip-support
TORCH_CUDA_ARCH_LIST="gfx942" pip install . --no-build-isolation

# Install Torch Sparse; MI300X corresponding architecture is gfx942
cd /workspace
git clone https://github.com/silogen/pytorch_sparse.git
cd pytorch_sparse
git checkout remotes/origin/feature/hip-support
git submodule update --init --recursive
TORCH_CUDA_ARCH_LIST="gfx942" pip install . --no-build-isolation

# Install MatterSim
cd /workspace
git clone https://github.com/microsoft/mattersim.git
cp /workspace/src/mattersim-pyproject.toml /workspace/mattersim/pyproject.toml
cp /workspace/src/batch_relax.patch /workspace/mattersim/batch_relax.patch
cd /workspace/mattersim
git apply batch_relax.patch
pip install -e .