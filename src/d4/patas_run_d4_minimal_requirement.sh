# this script install minimal requirements for D2 to run from preprocessed data files
if [ ! -d "./venv_condor" ]
then
  echo "Creating virtual environment..."
  /dropbox/22-23/575j/env/bin/python -m venv "./venv_condor"
  source ./venv_condor/bin/activate
  echo "Installing packages..."
  pip install numpy scikit-learn keras tensorflow pandas
fi

source ./venv_condor/bin/activate
python d4.py