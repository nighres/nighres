Installing Nighres
===================

From PyPI
----------

You can (soon!) install the latest stable release of nighres from PyPI::

    pip install nighres

From Github
------------

You can also get the latest version from Github ::

   git clone https://github.com/nighres/nighres

Or download and unpack the zip file from Github under **Clone and download** ->
**Download ZIP**

Once you have cloned or unpacked the nighres directory, you can use pip to set up the intallation ::

   cd nighres
   pip install .

Testing the installation
------------------------

You can often catch installation problems by simply import nighres in Python ::

    python -c "import nighres"

If that works, you can try running one of the examples. If you cloned the github repository, you can find them inside the subdirectory *examples*. Alternatively, you can also download the :ref:`examples <examples-index>` from the online documentation.

|
Troubleshooting
----------------

libjvm.so error
~~~~~~~~~~~~~~~~

You might get the following error when trying to import nighres::

    ImportError: libjvm.so: cannot open shared object file: No such file or directory

This is because the original CBS Tools Java code in the **cbstools** module has been compiled against a Java installation that is different from yours.

You can fix this by finding your libjvm.so location::

And then adding it to the library path. Depending on you Java installation it will be something similar to one of these::

    export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/
    export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64/server/

If that doesn't do the trick, try running::

    sudo R CMD javareconf

Rebuilding
~~~~~~~~~~~

If you the above does not work for you, you might have to
rebuild the package locally.

1. Make sure you have `JCC <http://jcc.readthedocs.io/en/latest/>`_ installed::

    sudo apt-get install jcc

2. Navigate to the nighres directory and run the build script::

    ./build.sh

The build might take a while because it pulls the original Java code from
https://github.com/piloubazin/cbstools-public, downloads its dependencies
*JIST* and *MIPAV*, compiles the Java classes and builds the wrappers using
JCC.

|
Dependencies
------------

.. todo:: Check and include version dependencies

Nighres depends on

* `numpy <http://www.numpy.org/>`_
* `nibabel <http://nipy.org/nibabel/>`_

These packages are automatically installed by pip when installing nighres.

The following dependencies are not necessary for running the built packages, so most likely you won't need to worry about them.

Building the documentation

* sphinx
* sphinx-rtd-theme
* sphinx-gallery
* matplotlib
* pillow

Building the packages

* JCC
