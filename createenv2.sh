eval "$(conda shell.bash hook)"
conda remove -n econ2 -y --all
conda create -n econ2 -y python=3.10
conda activate econ2

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install bitsandbytes transformers accelerate==0.21.0 datasets ftfy tensorboard Jinja2
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip uninstall diffusers -y
pip install --upgrade git+https://github.com/EnricoBeltramo/diffusers.git -q

conda install -c anaconda cython pip -y
conda install -c bottler nvidiacub -y
conda install -c conda-forge fvcore iopath pyembree -y
conda install -c conda-forge pytorch-lightning kornia -y
conda install -c pytorch3d pytorch3d -y

pip install cupy-cuda11x

pip install -r requirements2.txt

cd lib/common/libmesh
python setup.py build_ext --inplace
cd ../libvoxelize
python setup.py build_ext --inplace
cd ../../../

conda deactivate