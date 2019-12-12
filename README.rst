.. -*- mode: rst -*-

.. image:: https://travis-ci.org/nighres/nighres.svg?branch=master
    :target: https://travis-ci.org/nighres
    :alt: Travis CI build status
.. image:: https://readthedocs.org/projects/nighres/badge/?version=latest
    :target: http://nighres.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Nighres
=======

Nighres is a Python package for processing of high-resolution neuroimaging data.
It developed out of `CBS High-Res Brain Processing Tools
<https://www.cbs.mpg.de/institute/software/cbs-tools>`_ and aims to make those
tools easier to install, use and extend. Nighres now includes new functions from
the `IMCN imaging toolkit <https://github.com/IMCN-UvA/imcn-imaging>`_ and
additional packages may be added in future releases.

Because parts of the package have to be built locally it is currently not possible to
use ``pip install`` directly from PyPI. Instead, please follow the installation
instructions provided at http://nighres.readthedocs.io/en/latest/installation.html

Currently, Nighres is developed and tested on Linux Ubuntu Trusty and Python 3.5.
Release versions have been extensively tested on our example data. Development
versions include additional modules and functions but with limited guarantees.


Required packages
=================

In order to run Nighres, you will need:

* python >= 3.5
* Java JDK >= 1.7
* JCC >= 3.0

For instance, in Debian/Ubuntu (amd64 systems):

    sudo apt-get install openjdk-8-jdk
    export JCC_JDK=/usr/lib/jvm/java-8-openjdk-amd64
    python3 -m pip install jcc    (or just pip install jcc if Python 3 is your default)

For some functionalities you need extra packages

* Nipype & ANTs (for registration using ANTs)
* Pandas (for working with surface meshes)
* Nilearn (for plotting in the examples)

If you use nighres, please reference the publictation:

Huntenburg JM, Steele CJ, Bazin P-L (2018) Nighres: processing tools for high-resolution neuroimaging. Gigascience 7 Available at: https://academic.oup.com/gigascience/article/7/7/giy082/5049008

Docker
======

To quickly try out nighres in a preset, batteries-included environment, you can use the
included Dockerfile, which includes Ubuntu 14 Trusty, openJDK-8, nighres, and Jupyter
Notebook. The only thing you need to install is `Docker <https://www.docker.com/>`_, a
lightweight container platform that runs on Linux, Windows and Mac OS X.

To build the Docker image, do the following::

    git clone https://github.com/nighres/nighres
    cd nighres
    docker build . -t nighres

To run the Docker container::

    docker run --rm -p 8888:8888 nighres

Now go with your browser to https://localhost:8888 to start a notebook. You should be able
to import nighres by entering::

    import nighres

into the first cell of your notebook.

Usually you also want to have access to some data when you run nighres. You can grant the Docker container
access to a data folder on your host OS by using the `-v` tag when you start the container::

    docker run --rm -v /home/me/my_data:/data -p 8888:8888 nighres

Now, in your notebook you will be able to access your data on the path `/data`
