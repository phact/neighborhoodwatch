#!/bin/bash

echo
echo "=================================================================="
echo "== Install CUDA Toolkit 12.2"
echo "=================================================================="
echo
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update


echo "*** CUDA base installer ***"
echo 
sudo apt -y install cuda-12-2 cuda-toolkit-12-2

# Verify CUDA installation
echo
echo "*** Verifiy CUDA installation ***"
echo 
/usr/local/cuda/bin/nvcc --version

echo
echo "*** Install cuDNN library 8.9.6 ***"
sudo apt install -y libcudnn8=8.9.6.50-1+cuda12.2 libcudnn8-dev=8.9.6.50-1+cuda12.2 libcudnn8-samples=8.9.6.50-1+cuda12.2

# Verify cuDNN installation
echo
echo "*** Verifiy cuDNN installation ***"
sudo apt install -y zlib1g libfreeimage3 libfreeimage-dev
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN


# https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#down
echo
echo "*** Install NCCL library 2.19.2 ***"
echo
sudo apt install libnccl2=2.18.5-1+cuda12.2 libnccl-dev=2.18.5-1+cuda12.2

echo
echo
echo "=================================================================="
echo "== Install MiniConda                                            =="
echo "=================================================================="
echo
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
source ~/.bashrc

conda create -n py310 python=3.10
conda activate py310

echo
echo
echo "=================================================================="
echo "== Install Poetry and apache-beam                               =="
echo "=================================================================="
echo
# NOTE: can't manage 'apache-beam' with poetry due to unsolvable dependency conflict
pip install apache-beam
pip install poetry
echo "Make sure to run 'poetry lock && poetry install' in the project directory!"

echo "export PATH=\"/home/ubuntu/.local/bin:$PATH\"" >> ~/.bashrc
source ~/.bashrc