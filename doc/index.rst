.. nighres documentation master file, created by
   sphinx-quickstart on Wed Aug  2 19:13:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://travis-ci.org/nighres/nighres.svg?branch=master
   :target: https://travis-ci.org/nighres
   :alt: Travis CI build status (Linux Trusty)
.. image:: https://readthedocs.org/projects/nighres/badge/?version=latest
   :target: http://nighres.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

|


Welcome to Nighres!
====================

Nighres is a Python package for processing of high-resolution neuroimaging data.
It developed out of `CBS High-Res Brain Processing Tools
<https://www.cbs.mpg.de/institute/software/cbs-tools>`_ and aims to make those
tools easier to install, use and extend.

.. warning:: Nighres is currently still in beta stage

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Modules and Functions

   brain/index
   cortex/index
   surface/index
   laminar/index
   data/index
   io/index

.. toctree::
   :maxdepth: 2
   :caption: Good to know

   data_formats
   saving
   levelsets

.. toctree::
   :maxdepth: 1
   :caption: Developer's guide

   developers/index
   developers/setup
   developers/wrapping_cbstools
   developers/python_function
   developers/examples
   developers/docs
   developers/pr

|

.. admonition:: Credit

   Nighres is a community-developed project made possible by these `contributors <https://github.com/nighres/nighres/graphs/contributors>`_. The project was born and continues to evolve at `brainhack <http://www.brainhack.org/>`_.

   We thank the `Google Summer of Code 2017 <https://summerofcode.withgoogle.com/archive/>`_ and `INCF <https://www.incf.org/>`_ as a mentoring organization, for supporting the initial development phase of Nighres. See also the `development blog <https://juhuntenburg.github.io/gsoc2017/>`_.

   When using Nighres in your research, please make sure to cite the references mentioned in the documentation of the particular functions you use. We are also preparing a dedicated Nighres paper. For now, we suggest you cite:

   * Bazin et al. (2014) A computational framework for ultra-high resolution
     cortical segmentation at 7Tesla. `DOI: 10.1016/j.neuroimage.2013.03.077
     <http://www.sciencedirect.com/science/article/pii/S1053811913003327?via%3Dihub>`_
   * Huntenburg et al. (2017) Laminar Python: Tools for cortical
     depth-resolved analysis of high-resolution brain imaging data in
     Python. `DOI: 10.3897/rio.3.e12346 <https://riojournal.com/article/12346/>`_
