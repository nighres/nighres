Installing Nighres
===================

From source
-----------

Currently the only way to get Nighres is through the Github repository
https://github.com/nighres/nighres ::

   git clone https://github.com/nighres/nighres

Or download and unpack the zip file from Github under **Clone and download** ->
**Download ZIP**

Once you have cloned or unpacked the nighres directory ::

   cd nighres
   pip install .


Building the extensions
-----------------------

.. note:: You need to have `JCC <http://jcc.readthedocs.io/en/latest/>`_
   installed to build the extensions.

If you are not running on a common Linux distribution, you might have to
rebuild the **cbstools** module, which contains C++ wrappers of the original
CBS Tools Java code. Navigate to the main nighres directory and run the
build script::

    ./build.sh

The build might take a while because it pulls the original Java code from
https://github.com/piloubazin/cbstools-public, downloads its dependencies
*JIST* and *MIPAV*, compiles the Java classes and builds the wrappers using
JCC.

Dependencies
------------

.. todo:: Check and include version dependencies

Nighres depends on

* `numpy <http://www.numpy.org/>`_
* `nibabel <http://nipy.org/nibabel/>`_

These packages are automatically installed by pip when installing Nighres.

Additional dependencies for building the documentation (only necessary if you
are a developer and want to style the documentation)

* sphinx
* sphinx-rtd-theme
* sphinx-gallery
* matplotlib
* pillow
