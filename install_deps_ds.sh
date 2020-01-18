wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-ppc64le.sh
bash Anaconda3-5.2.0-Linux-ppc64le.sh -b

echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
echo '. $HOME/anaconda3/etc/profile.d/conda.sh' >> ~/.bashrc
# echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/software/cuda-9.2/targets/ppc64le-linux/lib/nvidia/:"' >> ~/.bashrc
echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.1/bin/:"' >> ~/.bashrc


export PATH="$HOME/anaconda3/bin:$PATH"
. $HOME/anaconda3/etc/profile.d/conda.sh

ENVNAME=echofilter
conda create -p "$HOME/venvs/$ENVNAME" -q python=3.6 pip
conda activate "$HOME/venvs/$ENVNAME"

conda install numpy pandas tqdm matplotlib seaborn jupyter pytorch torchvision tensorboard -c https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/ -y

pip install git+https://github.com/scottclowe/pytorch-utils#egg=torchutils

conda list

echo "$LD_LIBRARY_PATH"
# LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/software/cuda-9.2/targets/ppc64le-linux/lib/nvidia/:"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.1/bin/:"
echo "$LD_LIBRARY_PATH"

python -c "import torch; print(torch.__version__)"

