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

For some functionalities you need extra packages

* Nipype & ANTs (for registration using ANTs)
* Pandas (for working with surface meshes)
* Nilearn (for plotting in the examples)

For more information on how to install Nighres or use its docker version, please see
the `installation page of the documentation<https://nighres.readthedocs.io/en/latest/installation.html#installing-nighres>`_.

If you use nighres, please reference the publication:

Huntenburg JM, Steele CJ, Bazin P-L (2018) Nighres: processing tools for high-resolution neuroimaging. Gigascience 7 Available at: https://academic.oup.com/gigascience/article/7/7/giy082/5049008