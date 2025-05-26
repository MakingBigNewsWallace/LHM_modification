# install torch 2.4.0
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 

# install dependencies
pip install -r requirements.txt

# install from source code to avoid the conflict with torchvision
pip uninstall basicsr -y
pip install git+https://github.com/XPixelGroup/BasicSR

cd ..
# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install sam2
pip install git+https://github.com/hitsz-zuoqi/sam2/

# or
# git clone --recursive https://github.com/hitsz-zuoqi/sam2
# pip install ./sam2

# install diff-gaussian-rasterization
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/

# or
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization

# install simple-knn
pip install git+https://github.com/camenduru/simple-knn/

# or
# git clone https://github.com/camenduru/simple-knn.git
# pip install ./simple-knn
