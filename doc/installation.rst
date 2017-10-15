Installing Nighres
===================

From PyPI
----------

You can install the latest stable release of nighres from PyPI::

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

    find / -type f -name libjvm.so

And then adding it to the library path. Depending on you Java installation it will be something similar to one of these::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64/server/

This can be run within the current terminal for a single session, or made permanent by adding the export statement to your terminal execution script (i.e., .bashrc on most linux systems). If that doesn't do the trick, try running::

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

Nighres depends on:

* `Numpy <http://www.numpy.org/>`_
* `Nibabel <http://nipy.org/nibabel/>`_

These packages are automatically installed by pip when installing nighres.


.. _add-deps:

Additional dependencies (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting in the examples

* `Nilearn <http://nilearn.github.io/>`_ and its dependencies, if Nilearn is not installed, plotting in the examples will be skipped and you can view the results in any other nifti viewer

Building the Java extensions from source

* `JCC <http://jcc.readthedocs.io/en/latest/>`_

Building the documentation

* `sphinx <http://www.sphinx-doc.org/en/stable/>`_
* `sphinx-gallery <https://sphinx-gallery.github.io/>`_
* `matplotlib <http://matplotlib.org/>`_
* `sphinx-rtd-theme <http://docs.readthedocs.io/en/latest/theme.html>`_ (pip install sphinx-rtd-theme)
* `pillow <https://python-pillow.org/>`_ (pip install pillow)
