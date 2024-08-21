pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
git clone https://github.com/NASA-IMPACT/hls-foundation-os.git
cd hls-foundation-os
pip install -e .
pip install -U openmim
pip install ipykernel
pip install ipywidgets
pip install streamlit
mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.11.0/index.html
yes | pip uninstall numpy
pip install numpy==1.26.4
pip install huggingface_hub
pip install matplotlib
pip install scipy
pip install scikit-image