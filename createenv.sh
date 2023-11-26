eval "$(conda shell.bash hook)"
conda remove -n econ -y --all
conda create -n econ -y python=3.8
conda activate econ

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c anaconda cython pip -y
conda install -c bottler nvidiacub -y
conda install -c conda-forge fvcore iopath pyembree -y
conda install -c conda-forge pytorch-lightning kornia -y

pip install cupy-cuda116
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html

pip install -r requirements2.txt

cd lib/common/libmesh
python setup.py build_ext --inplace
cd ../libvoxelize
python setup.py build_ext --inplace
cd ../../../

conda deactivate