#!/usr/bin/env bash
# set -e -x

#
## SETUP
#

set -euo pipefail
unset CDPATH; cd "$( dirname "${BASH_SOURCE[0]}" )"; cd "$(pwd -P)"

fatal() { echo -e "$1"; exit 1; }
function join_by { local IFS="$1"; shift; echo "$*"; }

cbstools_repo="https://github.com/piloubazin/cbstools-public.git"
imcntk_repo="https://github.com/piloubazin/imcn-imaging.git"

release="release-1.4.0"

# Check the system has the necessary commands
hash wget tar javac jar python3 2>/dev/null || fatal "This script needs the following commands available: wget tar javac jar python3"

# Check for setuptools and wheels
pip_modules=$(python3 -m pip list | tr -s ' ' | cut -f 1 -d ' ')
echo "${pip_modules}" | grep setuptools > /dev/null || fatal 'This script requires setuptools.\nInstall with `python3 -m pip install --upgrade setuptools`'
echo "${pip_modules}" | grep wheel > /dev/null || fatal 'This script requires wheel.\nInstall with `python3 -m pip install --upgrade wheel`'

# echo "Before detection: $JAVA_HOME"

# Set the JAVA_HOME variable if it is not set
detected_home=$(java -XshowSettings:properties -version 2>&1 | tr -d ' '| grep java.home | cut -f 2 -d '=')
export JAVA_HOME=${JAVA_HOME:-"$detected_home"}
# echo "After detection: $JAVA_HOME"

# Check that JCC is installed
echo "${pip_modules}" | grep JCC > /dev/null || fatal 'This script requires JCC.\nInstall with `apt-get install jcc` or equivalent and `python3 -m pip install jcc`'

# Attempt to check for python development headers
# Inspired by https://stackoverflow.com/a/4850603
python_include_path=$(python3 -c "from distutils import sysconfig as s; print(s.get_config_vars()['INCLUDEPY'])")
test -f "${python_include_path}/Python.h" || fatal 'This script requires python development headers.\nInstall with `apt-get install python-dev`, or \n `apt-get install python3-dev`, or equivalent'

#
## COMPILE CBSTOOLS
#

# Get cbstools git clone
test -d cbstools-public && (
    cd cbstools-public
	#git checkout $release
	git checkout master
	git pull
	cd ..
) || (
	git clone $cbstools_repo
	cd cbstools-public
	#git checkout $release
	git checkout master
	git pull
	cd ..
)

# Java dependencies. Order matters
deps=(
	"."
	"lib/Jama-mipav.jar"
	"lib/commons-math3-3.5.jar"
)
deps_list=$(join_by ":" "${deps[@]}")

# List of library files needed to run the cbstools core functions
source cbstools-lib-files.sh
echo $cbstools_files # result is in $cbstools_files

cbstools_list=$(join_by " " "${cbstools_files[@]}")

# Compilation options
javac_opts=(
	# "-d build"        # Output dir
	"-Xlint:none"     # Disable all warnings
	# "-server"         # ?
	"-g"              # Generate all debugging info
	"-O"              # ?
	"-deprecation"    # Show information about deprecated Java calls
	"-encoding UTF-8" # Require UTF-8, rather than platform-specifc
)

echo "Compiling..."
cd cbstools-public
javac -cp ${deps_list} ${javac_opts[@]} ca/concordia/qpi/*/*.java de/mpg/cbs/core/*/*.java $cbstools_list


echo "Assembling..."
mkdir -p ../nighresjava/src
mkdir -p ../nighresjava/lib

#jar cf cbstools.jar     de/mpg/cbs/core/*/*.class
jar cf ../nighresjava/src/nighresjava.jar de/mpg/cbs/core/*/*.class ca/concordia/qpi/*/*.class
jar cf ../nighresjava/src/cbstools-lib.jar de/mpg/cbs/*/*.class

cp lib/*.jar ../nighresjava/lib/
cd ..

#
## COMPILE IMCNTK
#

# Get imcntk git clone
test -d imcn-imaging && (
    cd imcn-imaging
	git checkout master
	git pull
	cd ..
) || (
	git clone $imcntk_repo
	cd imcn-imaging
	git checkout master
	git pull
	cd ..
)

# Java dependencies. Order matters
deps=(
	"."
	"lib/Jama-mipav.jar"
	"lib/commons-math3-3.5.jar"
)
deps_list=$(join_by ":" "${deps[@]}")

# List of library files needed to run the cbstools core functions
source imcntk-lib-files.sh
echo $imcntk_files # result is in $cbstools_files

imcntk_list=$(join_by " " "${imcntk_files[@]}")

# Compilation options
javac_opts=(
	# "-d build"        # Output dir
	"-Xlint:none"     # Disable all warnings
	# "-server"         # ?
	"-g"              # Generate all debugging info
	"-O"              # ?
	"-deprecation"    # Show information about deprecated Java calls
	"-encoding UTF-8" # Require UTF-8, rather than platform-specifc
)

echo "Compiling..."
cd imcn-imaging
javac -cp ${deps_list} ${javac_opts[@]} nl/uva/imcn/algorithms/*.java $imcntk_list

echo "Assembling..."
jar uf ../nighresjava/src/nighresjava.jar nl/uva/imcn/algorithms/*.class
jar cf ../nighresjava/src/imcntk-lib.jar nl/uva/imcn/libraries/*.class nl/uva/imcn/methods/*.class nl/uva/imcn/structures/*.class nl/uva/imcn/utilities/*.class

cp lib/*.jar ../nighresjava/lib/
cd ..

#
## WRAP TO PYTHON
#
cd nighresjava

jcc_args=(
	# All public methods in this JAR will be wrapped
	"--jar src/nighresjava.jar"

	# Dependencies
	"--include src/cbstools-lib.jar"
	"--include src/imcntk-lib.jar"
	"--include lib/commons-math3-3.5.jar"
	"--include lib/Jama-mipav.jar"

	# Name the python module
	"--python nighresjava"

	# Java VM heap size limit
	"--maxheap 4096M"

	# Compile
	"--build"
)

python3 -m jcc ${jcc_args[@]}

# post-processing for Mac users: replace __dir__ variable in __init__.py
sed -i -e 's/__dir__/__ndir__/g' build/nighresjava/__init__.py
#
# Assemble PYPI package
#

echo "Copying necessary files for nires pypi package..."

cd ..
#cp -rv build/nighresjava/ ../
cp -rv nighresjava/build/nighresjava/* nighresjava/

# Find and copy the shared object file for the current architecture
#find build/ -type f | grep '.so$' | head -n 1 | xargs -I '{}' -- cp '{}' ../nighresjava/_nighresjava.so
find nighresjava/build/ -type f | grep '.so$' | head -n 1 | xargs -I '{}' -- cp '{}' nighresjava/_nighresjava.so
#cd ..

echo ""
echo "======="
echo "You should now be able to install nighres with pip"
echo " python3 -m pip install ."
echo "======="
echo ""

#python3 -m pip install .

# Make the python wheel
# PLT=$(uname | tr '[:upper:]' '[:lower:]')
# for now use manylinux
# PLT="manylinux1"
# CPU=$(lscpu | grep -oP 'Architecture:\s*\K.+')
# PY="$(python -V 2>&1)"
# if [[ $PY == *2\.*\.* ]]; then
#     python setup.py bdist_wheel --dist-dir dist --plat-name ${PLT}_${CPU} --python-tag py2
# elif [[ $PY == *3\.*\.* ]]; then
# 	python setup.py bdist_wheel --dist-dir dist --plat-name ${PLT}_${CPU} --python-tag py3
# fi


# remove unused folders (optional, requires re-download
#rm -rf cbstools-public
#rm -rf imcn-imaging
