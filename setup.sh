#!/bin/bash

set -euo pipefail

# BindCraft setup script (conda env + uv pip)
# - Creates a conda env with Python 3.12
# - Installs system/binary packages via conda (PyRosetta, ffmpeg, optional CUDA)
# - Installs Python packages via uv pip (fast resolver/installer)
# - Downloads AlphaFold2 model weights
# - Sets executable permissions on included tools

# conda create -n bindcraft python=3.12 --yes
# conda activate bindcraft

mamba install -c conda-forge libgfortran5 ffmpeg --yes
mamba install -c https://conda.graylab.jhu.edu pyrosetta --yes
mamba install -c conda-forge pdbfixer --yes

pip install uv

uv pip install jax[cuda12]==0.6.2

# Base scientific stack
UV_PKGS=(
    biopython
    chex
    dm-haiku
    dm-tree
    flax
    fsspec
    immutabledict
    joblib
    jupyter
    matplotlib
    ml-collections
    numpy
    optax
    pandas
    py3dmol
    scipy
    seaborn
    tqdm
)

# Install the rest
uv pip install "${UV_PKGS[@]}"

# Make sure all required packages were installed
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

install_dir=$(pwd)

# install ColabDesign
echo -e "Installing ColabDesign\n"
uv pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps || {
    echo -e "Error: Failed to install ColabDesign"
    exit 1
}
python -c "import colabdesign" >/dev/null 2>&1 || {
    echo -e "Error: colabdesign module not found after installation"
    exit 1
}

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || {
    echo -e "Error: Failed to create weights directory"
    exit 1
}
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || {
    echo -e "Error: Failed to download AlphaFold2 weights"
    exit 1
}
[ -s "${params_file}" ] || {
    echo -e "Error: Could not locate downloaded AlphaFold2 weights"
    exit 1
}

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || {
    echo -e "Error: Corrupt AlphaFold2 weights download"
    exit 1
}
tar -xvf "${params_file}" -C "${params_dir}" || {
    echo -e "Error: Failed to extract AlphaFold2weights"
    exit 1
}
[ -f "${params_dir}/params_model_5_ptm.npz" ] || {
    echo -e "Error: Could not locate extracted AlphaFold2 weights"
    exit 1
}
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || {
    echo -e "Error: Failed to chmod dssp"
    exit 1
}
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || {
    echo -e "Error: Failed to chmod DAlphaBall.gcc"
    exit 1
}

# finish
# conda deactivate
echo -e "BindCraft environment set up\n"
