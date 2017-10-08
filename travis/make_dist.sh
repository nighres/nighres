#!/bin/bash
set -e -x

pwd
# create the source distribution
python setup.py sdist

# make a new folder to test the installation
mkdir "$HOME"/test_install
tar -xf dist/nighres* -C "$HOME"/test_install
cd "$HOME"/test_install/nighres*
./build.sh
pip install --user .
cd "$HOME"

ls virtualenv

# Run test
python "$NIGHRES_DIR"/examples/example_tissue_classification.py
