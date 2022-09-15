PROJECT_DIRECTORY=$(pwd)

echo "setting up python virtual environment"
mkdir -p 'affordances_venv'
python3 -m venv ./affordances_venv

echo "sourcing venv and installing requirements"
source ./affordances_venv/bin/activate
pip3 install -r requirements.txt

echo "installing pytorch and network architecture"
pip3 install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .

echo "python dependencies were installed successfully"

