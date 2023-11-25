eval "$(conda shell.bash hook)"
conda remove -n econ -y --all
conda create -n econ -y python=3.10
conda activate econ

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c bottler nvidiacub -y
conda install -c conda-forge fvcore iopath pyembree -y
conda install -c anaconda cupy cython pip -y

pip install -r requirements.txt

conda deactivate