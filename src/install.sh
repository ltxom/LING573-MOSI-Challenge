if [ ! -d "../venv" ]
then
  echo "Creating virtual environment..."
  python3 -m venv "../venv"
  source ../venv/bin/activate
  echo "Installing packages..."
  pip install h5py validators tqdm numpy argparse requests colorama
fi
source ../venv/bin/activate

echo "Cloning the repository for SDK..."
git clone https://github.com/dearbornlavern/CMU-MultimodalSDK.git
export PYTHONPATH="$PWD/CMU-MultimodalSDK:$PYTHONPATH"
pip install notebook scikit-learn keras tensorflow matplotlib pandas