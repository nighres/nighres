#!/bin/bash
set -e -x

# create the source distribution
python3 setup.py sdist

# make a new folder to test the installation
mkdir "$HOME"/test_install
tar -xf dist/nighres* -C "$HOME"/test_install
cd "$HOME"/test_install/nighres*
./build.sh
python3 -m pip install --user .
cd "$HOME"

# Run test
#python "$HOME"/build/nighres/nighres/examples/example_tissue_classification.py
