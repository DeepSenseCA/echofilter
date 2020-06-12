# Download and install Anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-ppc64le.sh
bash Anaconda3-5.2.0-Linux-ppc64le.sh -b

# Set up .bashrc file to add anaconda and cuda to path env variables
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
echo '. $HOME/anaconda3/etc/profile.d/conda.sh' >> ~/.bashrc
# echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/software/cuda-9.2/targets/ppc64le-linux/lib/nvidia/:"' >> ~/.bashrc
echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.1/bin/:"' >> ~/.bashrc

# And set up the env variables for the current session
export PATH="$HOME/anaconda3/bin:$PATH"
. $HOME/anaconda3/etc/profile.d/conda.sh
# LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/software/cuda-9.2/targets/ppc64le-linux/lib/nvidia/:"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.1/bin/:"

# Make a new virtual environment with conda
ENVNAME=echofilter
conda create -p "$HOME/venvs/$ENVNAME" -q python=3.6 pip
conda activate "$HOME/venvs/$ENVNAME"

conda install numpy scipy pandas tqdm matplotlib seaborn jupyter apex pytorch torchvision tensorboard tensorflow pillow scikit-image -c https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/ -y

pip install git+https://github.com/scottclowe/pytorch-utils
pip install git+https://github.com/davidtvs/pytorch-lr-finder.git@1d1e3e7170db3d784ac7e268a63fa8a077880b50
pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git@8d636a560b71e740e9423aa10ba4c7f751bb9120
pip install -e .

conda list

echo "$LD_LIBRARY_PATH"

python -c "import torch; print(torch.__version__)"
