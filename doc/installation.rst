Installing Nighres
===================

From PyPI
----------

You can download the latest stable release of Nighres from `PyPI <https://pypi.python.org/pypi/nighres>`_.

Because parts of the package have to be built locally it is currently not possible to use ``pip install`` directly from PyPI. Instead, please download and unpack the tarball to :ref:`build-nighres`. (Or use the :ref:`Docker image <docker-image>`)

From Github
------------

You can also get the latest version from Github ::

   git clone https://github.com/nighres/nighres

Or download and unpack the zip file from Github under **Clone and download** ->
**Download ZIP**

.. _build-nighres:

Build Nighres
--------------
1. Make sure you have `JCC <http://jcc.readthedocs.io/en/latest/>`_ installed. You need to both get the package via your package manager and install using pip to make it accessible to the Python interpreter, e.g::

    sudo apt-get install jcc
    pip install jcc

2. Navigate to the Nighres directory you downloaded (and unpacked) and run the build script::

    ./build.sh

The build might take a while because it pulls the original Java code from
https://github.com/piloubazin/cbstools-public and builds the wrappers using
JCC.

3. Install the Python package::

    pip install .


Testing the installation
------------------------

You can often catch installation problems by simply import Nighres in Python. Make sure to navigate out of the directory from which you installed to make sure Nighres has actually been installed correctly and can be accessed from any location ::

    python -c "import nighres"

If that works, you can try running one of the examples. You can find them inside the unpacked Nighres directory, in the subdirectory *examples*. Alternatively, you can also download the :ref:`examples <examples-index>` from the online documentation.

.. |
..
.. Troubleshooting
.. ----------------
..
.. libjvm.so error
.. ~~~~~~~~~~~~~~~~
..
.. You might get the following error when trying to import nighres::
..
..     ImportError: libjvm.so: cannot open shared object file: No such file or directory
..
.. This is because the original CBS Tools Java code in the **cbstools** module has been compiled against a Java installation that is different from yours.
..
.. You can fix this by finding your libjvm.so location::
..
..     find / -type f -name libjvm.so
..
.. And then adding it to the library path. Depending on you Java installation it will be something similar to one of these::
..
..     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/
..     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64/server/
..
.. This can be run within the current terminal for a single session, or made permanent by adding the export statement to your terminal execution script (i.e., .bashrc on most linux systems).
..
.. If that doesn't do the trick, try running::
..
..     sudo R CMD javareconf
..
.. Rebuilding
.. ~~~~~~~~~~~
..
.. If you the above does not work for you, you might have to
.. rebuild the package locally.
..
.. 1. Make sure you have `JCC <http://jcc.readthedocs.io/en/latest/>`_ installed::
..
..     sudo apt-get install jcc
..
.. 2. Navigate to the nighres directory and run the build script::
..
..     ./build.sh
..
.. The build might take a while because it pulls the original Java code from
.. https://github.com/piloubazin/cbstools-public, downloads its dependencies
.. *JIST* and *MIPAV*, compiles the Java classes and builds the wrappers using
.. JCC.
..
.. |

Dependencies
------------
To build Nighres you need:

* `JCC <http://jcc.readthedocs.io/en/latest/>`_
* Java, a version that is 1.7 or higher

To run Nighres depends on:

* `Numpy <http://www.numpy.org/>`_
* `Nibabel <http://nipy.org/nibabel/>`_

If not already available, these packages are automatically installed when you run pip install.

.. todo:: Check and include version dependencies


.. _docker-image:

Docker
------
To quickly try out nighres in a preset, batteries-included environment, you can use the included Dockerfile, which includes Ubuntu 14 Trusty, openJDK-8, nighres, and Jupyter Notebook. The only thing you need to install is `Docker <https://www.docker.com/>`_, a lightweight container platform that runs on Linux, Windows and Mac OS X.

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


.. _add-deps:

Additional dependencies (optional)
----------------------------------

Plotting in the examples

* `Nilearn <http://nilearn.github.io/>`_ and its dependencies, if Nilearn is not installed, plotting in the examples will be skipped and you can view the results in any other nifti viewer

Building the documentation

* `sphinx <http://www.sphinx-doc.org/en/stable/>`_
* `sphinx-gallery <https://sphinx-gallery.github.io/>`_
* `matplotlib <http://matplotlib.org/>`_
* `sphinx-rtd-theme <http://docs.readthedocs.io/en/latest/theme.html>`_ (pip install sphinx-rtd-theme)
* `pillow <https://python-pillow.org/>`_ (pip install pillow)

Using the docker image

* `Docker <https://www.docker.com/>`_
