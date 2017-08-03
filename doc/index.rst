.. nighres documentation master file, created by
   sphinx-quickstart on Wed Aug  2 19:13:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Nighres
========

.. container:: doc

   .. admonition:: Getting started

      .. toctree::
          :maxdepth: 1

          installation

   .. admonition:: Modules

      .. hlist::
         :columns: 1

         * Input/Output

           .. toctree::
              :maxdepth: 2
              :glob:

              io/*

         * Brain

           .. toctree::
              :maxdepth: 2
              :glob:

              brain/*

         * Laminar

           .. toctree::
              :maxdepth: 2
              :glob:

              laminar/*

         * Surface

           .. toctree::
              :maxdepth: 2
              :glob:

              surface/*

   .. admonition:: Examples

      Check out the  `example pipeline <https://github.com/nighres/nighres/blob/master/nighres/example_pipeline.py>`_.


.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
