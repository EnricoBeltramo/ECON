
call conda remove -n econ2 -y --all
call conda create -n econ2 -y python=3.10
call conda activate econ2

call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
call pip install transformers tomli
call pip install --upgrade git+https://github.com/EnricoBeltramo/diffusers.git 
call conda install -c conda-forge opencv -y



call conda deactivate