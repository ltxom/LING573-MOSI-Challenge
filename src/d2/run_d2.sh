# this file install all the packages that required to run D2 from scratch
if [ ! -d "./venv" ]
then
  echo "Creating virtual environment..."
  python3 -m venv "./venv"
  source ./venv/bin/activate
  echo "Installing packages..."
  pip install h5py validators tqdm numpy argparse requests colorama scikit-learn keras tensorflow matplotlib pandas
fi
source ./venv/bin/activate

echo "Cloning the repository for SDK..."
git clone https://github.com/dearbornlavern/CMU-MultimodalSDK.git
export PYTHONPATH="$PWD/CMU-MultimodalSDK:$PYTHONPATH"

#python download_data.py
python d2.py