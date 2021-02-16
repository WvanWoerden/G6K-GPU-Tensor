******************************
G6K - GPU Tensor
******************************

G6K is an open-source C++ and Python (2) library that implements several Sieve algorithms to be used in more advanced lattice reduction tasks. It follows the stateful machine framework from: 

Martin R. Albrecht and Léo Ducas and Gottfried Herold and Elena Kirshanova and Eamonn W. Postlethwaite and Marc Stevens, 
The General Sieve Kernel and New Records in Lattice Reduction.

The main source is available in `fplll/g6k <https://github.com/fplll/g6k>`__

This fork expands the G6K implementation with GPU, and in particular Tensor Core, accelerated sieves, and is accompanied by the work:

Léo Ducas, Marc Stevens, Wessel van Woerden,
Advanced Lattice Sieving on GPUs, with Tensor Cores, 
Eurocrypt 2021 (`eprint <https://eprint.iacr.org/2021/141.pdf>`__).

Note the this fork has been expanded from a `pretty old commit <https://github.com/fplll/g6k/commit/11e202967bf16ce5fe40258597fed54849e10a69>`__.

The CPU-only version of the BDGL-like sieve has been integrated into the `main g6k repository <https://github.com/fplll/g6k>`__, with further improvements, and we aim for long term maintenance. 
The GPU implementation has been made public in this repository, but with a lower commitment to quality, documentation and maintenance. Nevertheless feel free to create issues in this repository.

Building the library
====================

The code has only been tested on the NVIDIA Turing generation, and might not work on more recent GPUs.

You will need the current master of FPyLLL and a recent version of the CUDA Toolkit. See ``bootstrap.sh`` for creating all dependencies from scratch except for the CUDA Toolkit:

.. code-block:: bash

    ./bootstrap.sh                # once only: creates local python env, builds fplll, fpylll and G6K
    source g6k-env/bin/activate   # for every new shell: activates local python env
    ./rebuild.sh -f -y            # whenever you want to rebuild G6K

Otherwise, you will need fplll and fpylll already installed and build the G6K Cython extension **in place** like so:

.. code-block:: bash

    pip install Cython
    pip install -r requirements.txt
    ./rebuild.sh -f -y

Remove ``-f`` option to compile faster (fewer optimisations). 
The ``-y`` option significantly reduces the memory footprint, but disables the standard cpu-only sieves. See ``rebuild.sh`` for more options.


Code examples
=============

You can run a single svp-challenge instance on a multiple cores and multiple GPUs, for example:

.. code-block:: bash

    ./svp_challenge.py 100 --threads 4 --gpus 1 --verbose

Will run a svp-challenge using 4 CPU threads and a single GPU.

For more details on the parameters used for the `SVP records <https://www.latticechallenge.org/svp-challenge/halloffame.php>`__ see Section 7.2 of the `paper <https://eprint.iacr.org/2021/141.pdf>`__ or ``runchal2.sh``.

BDGL-sieve
----------

The BDGL-like GPU sieve can be enabled by running

.. code-block:: bash

    ./svp_challenge.py 100 --threads 4 --gpus 1 --gpu_bucketer bdgl --verbose

