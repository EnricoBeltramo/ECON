
call conda remove -n econsegmentationhuman3 -y --all
call conda create -n econsegmentationhuman3 -y python=3.10
call conda activate econsegmentationhuman3
call conda install -c pytorch pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -y
call pip install opencv-python
call pip install scikit-image


call pip install -r  requirementssegmentation.txt
call pip install git+https://github.com/openai/CLIP.git