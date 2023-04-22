# this script install minimal requirements for D2 to run from preprocessed data files
if [ ! -d "./venv_condor" ]
then
  echo "Creating virtual environment..."
  python3 -m venv "./venv_condor"
  source ./venv_condor/bin/activate
  echo "Installing packages..."
  pip install numpy scikit-learn keras tensorflow
fi

python d2.py