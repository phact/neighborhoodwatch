#!/bin/bash

## 
#  Install CUDA toolkit 12.3 
#  * https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
#
echo "=================================================================="
echo "== Install CUDA Toolkit 12.3                                    =="
echo "=================================================================="
echo
echo "== CUDA base installer ..."
echo 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install cuda cuda-toolkit-12-3

#  > driver installer (open kernel module flavor)
echo
echo "=> CUDA driver installer (open kernel module) ..."
echo 
sudo apt install -y nvidia-kernel-open-545
sudo apt install -y cuda-drivers-545

# Verify CUDA installation
echo
echo "=> Verifiy CUDA installation ..."
echo 
/usr/local/cuda/bin/nvcc --version

echo
echo
echo

##
#  Install cuDNN library 8.9.6
#  * https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
#  * https://developer.nvidia.com/rdp/cudnn-download
#
echo "=================================================================="
echo "== Install cuDNN library 8.9.6                                  =="
echo "=================================================================="
echo
echo "=> Please make sure you download the cuDNN library package file (.deb) from the following"
echo "   login-protected web page: https://developer.nvidia.com/rdp/cudnn-download"
echo
if [[ ! -f cudnn-local-repo-ubuntu2204-8.9.6.50_1.0-1_amd64.deb ]]; then
  echo "Can't find the cuDNN installation package file!"
  exit 1
fi
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.6.50_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn8=8.9.6.50-1+cuda12.2 libcudnn8-dev=8.9.6.50-1+cuda12.2 libcudnn8-samples=8.9.6.50-1+cuda12.2

# Verify cuDNN installation
echo
echo "=> Verifiy cuDNN installation ..."
echo
sudo apt install -y zlib1g libfreeimage3 libfreeimage-dev
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN

echo
echo
echo

##
#  NCCL install library 2.19.3
#  * https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian
#  * https://developer.nvidia.com/nccl/nccl-download
#
echo "=================================================================="
echo "== Install NCCL library 2.19.3                                  =="
echo "=================================================================="
echo
echo "=> Please make sure you download the NCCL library package file (.deb) from the following"
echo "   login-protected web page: https://developer.nvidia.com/nccl/nccl-download"
echo
if [[ ! -f nccl-local-repo-ubuntu2004-2.19.3-cuda12.3_1.0-1_amd64.deb ]]; then
  echo "Can't find the NCCL installation package file!"
  exit 1
fi
sudo dpkg -i nccl-local-repo-ubuntu2004-2.19.3-cuda12.3_1.0-1_amd64.deb
sudo cp /var/nccl-local-repo-ubuntu2004-2.19.3-cuda12.3/nccl-local-52155FEE-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libnccl2=2.19.3-1+cuda12.3 libnccl-dev=2.19.3-1+cuda12.3

# Verify NCCL installation
# echo
# echo "=> Verifiy NCCL installation ..."
# echo
# cd $HOME
# git clone https://github.com/NVIDIA/nccl-tests.git
# cd nccl-tests
# make

echo
echo
echo

##
#  Install MiniConda
#  * https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install
#
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
echo

##
#  Install Poetry
#  * https://python-poetry.org/docs/#installing-with-the-official-installer
#
echo
echo "=================================================================="
echo "== Install Poetry                                               =="
echo "=================================================================="
echo
curl -sSL https://install.python-poetry.org | python3 -

echo "export PATH=\"/home/ubuntu/.local/bin:$PATH\"" >> ~/.bashrc
source ~/.bashrc