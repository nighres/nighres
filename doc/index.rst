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

.. danger:: Nighres is currently in beta stadium and not fully functional

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Modules and Functions

   brain/index
   surface/index
   laminar/index
   data/index
   io/index

.. toctree::
   :maxdepth: 2
   :includehidden:
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Good to know

   data_formats
   saving
   levelsets

|

.. admonition:: Reference

   Until a dedicated Nighres paper is out, please cite the following references:

   * Bazin et al. (2014) A computational framework for ultra-high resolution
     cortical segmentation at 7Tesla. `DOI: 10.1016/j.neuroimage.2013.03.077
     <http://www.sciencedirect.com/science/article/pii/S1053811913003327?via%3Dihub>`_
   * Huntenburg et al. (2017) Laminar Python: Tools for cortical
     depth-resolved analysis of high-resolution brain imaging data in
     Python. `DOI: 10.3897/rio.3.e12346 <https://riojournal.com/article/12346/>`_
