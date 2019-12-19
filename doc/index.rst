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
tools easier to install, use and extend. Nighres now includes new functions from
the `IMCN imaging toolkit <https://github.com/IMCN-UvA/imcn-imaging>`_.

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
   data/index
   filtering/index
   intensity/index
   io/index
   laminar/index
   microscopy/index
   registration/index
   segmentation/index
   shape/index
   statistics/index
   surface/index



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

.. admonition:: Reference

   Huntenburg, Steele \& Bazin (2018). Nighres: processing tools for high-resolution neuroimaging. GigaScience, 7(7). `https://doi.org/10.1093/gigascience/giy082 <https://doi.org/10.1093/gigascience/giy082>`_


   Make sure to also cite the references indicated for the particular functions you use!


.. admonition:: Credit

   Nighres is a community-developed project made possible by its `contributors <https://github.com/nighres/nighres/graphs/contributors>`_. The project was born and continues to evolve at `brainhack <http://www.brainhack.org/>`_. We thank the `Google Summer of Code 2017 <https://summerofcode.withgoogle.com/archive/>`_ and `INCF <https://www.incf.org/>`_ as a mentoring organization, for supporting the initial development phase of Nighres. See also the `development blog <https://juhuntenburg.github.io/gsoc2017/>`_.
