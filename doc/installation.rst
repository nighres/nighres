Installing Nighres
===================

Requirements
------------

To build Nighres you need:

* Python 3.5 or higher
* `JCC <http://jcc.readthedocs.io/en/latest/>`_ 3
* Java JDK 1.7 or higher


The following Python packages are automatically installed with Nighres

* `Numpy <http://www.numpy.org/>`_
* `Nibabel <http://nipy.org/nibabel/>`_
* `psutils <https://pypi.org/project/psutil/>`_

For further dependencies of specific interfaces see :ref:`add-deps`.

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
1. Make sure you have Java JDK and JCC installed and set up. You will likely need to point the JCC_JDK variable to you Java JDK installation, e.g on a Debian/Ubuntu amd64 system::

    sudo apt-get install openjdk-8-jdk
    export JCC_JDK=/usr/lib/jvm/java-8-openjdk-amd64
    pip install jcc

2. Navigate to the Nighres directory you downloaded (and unpacked) and run the build script::

    ./build.sh

If you experience errors regarding missing java libraries (such as ljvm/libjvm or ljava/libjava), although you install Java JDK, it can be that JCC does not find the libraries for some reason. It can help to search for the "missing" library and make a symbolic link to it like this::

    sudo find / -type f -name libjvm.so   (this returns e.g. /usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so)
    sudo ln -s /usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so /usr/lib/libjvm.so


3. Install the Python package::

    pip install .


Testing the installation
------------------------

You can often catch installation problems by simply import Nighres in Python. Make sure to navigate out of the directory from which you installed to make sure Nighres has actually been installed correctly and can be accessed from any location ::

    python -c "import nighres"

If that works, you can try running one of the examples. You can find them inside the unpacked Nighres directory, in the subdirectory *examples*. Alternatively, you can also download the :ref:`examples <examples-index>` from the online documentation.


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

Optional dependencies
----------------------

Working with surface mesh files

* `pandas <https://pandas.pydata.org/>`_

Using the registration tools

* `nipype <https://nipype.readthedocs.io/en/latest/>`_
* `ANTs <https://github.com/ANTsX/ANTs>`_

Plotting in the examples

* `Nilearn <http://nilearn.github.io/>`_ and its dependencies, if Nilearn is not installed, plotting in the examples will be skipped and you can view the results in any other nifti viewer

Using the docker image

* `Docker <https://www.docker.com/>`_

Building the documentation

* `sphinx <http://www.sphinx-doc.org/en/stable/>`_
* `sphinx-gallery <https://sphinx-gallery.github.io/>`_
* `matplotlib <http://matplotlib.org/>`_
* `sphinx-rtd-theme <http://docs.readthedocs.io/en/latest/theme.html>`_ (pip install sphinx-rtd-theme)
* `pillow <https://python-pillow.org/>`_ (pip install pillow)
* `mock <https://pypi.org/project/mock/>`_
