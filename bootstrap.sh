#!/usr/bin/env bash

PYTHON=python2 # lducas/g6k/gpu_no_yr requires python2

jobs="-j 4 "
if [ "$1" = "-j" ]; then
	jobs="-j $2 "
fi

# Create Virtual Environment

if [ "$PYTHON" = "" ]; then
	PYTHON=python
	export PYTHON
fi
PIP="$PYTHON -m pip"
PYVER=$($PYTHON --version | cut -d' ' -f2)
echo "Usage:"
echo "   ./bootstrap.sh [ -j <#jobs> ]  (uses system's python)"
echo "   PYTHON=python2 ./bootstrap.sh  (uses python2)"
echo "   PYTHON=python3 ./bootstrap.sh  (uses python3)"
echo " "
echo "Using python version: $PYVER"
echo "Using $jobs"
sleep 1


rm -rf g6k-env
$PYTHON -m virtualenv g6k-env
cat <<EOF >>g6k-env/bin/activate
### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK

### PKG_CONFIG_HACK
_OLD_PKG_CONFIG_PATH="\$PKG_CONFIG_PATH"
PKG_CONFIG_PATH="\$VIRTUAL_ENV/lib/pkgconfig:\$PKG_CONFIG_PATH"
export PKG_CONFIG_PATH
### END_PKG_CONFIG_HACK

### PYTHON_ENV
PYTHON="$PYTHON"
export PYTHON
unalias python 2>/dev/null
unalias pip 2>/dev/null
alias python="$PYTHON"
alias pip="$PYTHON -m pip"

### prevent numpy from using multiple threads
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
EOF

if [ ! -d g6k-env ]; then
	echo "Failed to create virtual environment in 'g6k-env' !"
	echo "Is '$PYTHON -m virtualenv' working?"
	echo "Try '$PYTHON -m pip install virtualenv' otherwise."
	exit 1
fi

ln -s g6k-env/bin/activate ./
source g6k-env/bin/activate

$PIP install -U pip
$PIP install Cython
$PIP install cysignals

cat <<EOF >>g6k-env/bin/activate
CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS
EOF

deactivate
source g6k-env/bin/activate


# Install FPLLL

git clone https://github.com/fplll/fplll g6k-fplll
cd g6k-fplll || exit 1
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS
make clean
make $jobs
make install || exit 1
cd ..

# Install FPyLLL
git clone https://github.com/fplll/fpylll g6k-fpylll
cd g6k-fpylll || exit 1
$PIP install Cython
$PIP install -r requirements.txt
$PIP install -r suggestions.txt
$PYTHON setup.py clean
$PYTHON setup.py build_ext $jobs || $PYTHON setup.py build_ext
$PYTHON setup.py install || exit 1
cd ..

# Install parallel-hashmap
[ -d parallel-hashmap ] || git clone https://github.com/cr-marcstevens/parallel-hashmap


$PIP install -r requirements.txt
$PYTHON setup.py clean
$PYTHON setup.py build_ext $jobs --inplace || $PYTHON setup.py build_ext --inplace

echo " "
echo "Don't forget to activate environment each time:"
echo " source ./activate"
echo "This will also add the following aliases:"
grep "^alias" activate

