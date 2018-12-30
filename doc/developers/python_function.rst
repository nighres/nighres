.. _python-function:

Adding a new Python function
=============================
Great, you want to contribute a Python function or module to Nighres!

Before you begin
-----------------

Please take a look at the code that is `under development <https://github.com/nighres/nighres/pulls>`_ or `discussion <https://github.com/nighres/nighres/issues>`_. If you see something related get in touch with the developer(s) directly to combine your efforts.

Take a moment to familiarise yourself with the documentation and functions, it is likely that some helper functions that you will want to use (e.g., `I/O <http://nighres.readthedocs.io/en/latest/io/index.html>`_) have already been coded and tested.

Let's get into it
-----------------

1. Follow the steps described in :ref:`set-up`

2. Decide  where to put your code

   If the functionality that you are adding falls within the scope of a pre-existing submodule, edit or create files within the submodule. If it is entirely new (e.g., *statistics*), create a new submodule with its own dedicated subdirectory.

3. Coding
   If you are creating a submodule from scratch, an easy way to start is to copy the *__init.py__* and initial import statements from an existing module.

   Please code `PEP8 compliant <https://www.python.org/dev/peps/pep-0008/>`_, best to use a Python linter in your editor.

   Please keep within our :ref:`documentation guidelines <adapt-docs>`.

   Leave plenty of comments in your code so that others can understand and possibly adapt your code later.

   Use the standard `I/O interfaces <http://nighres.readthedocs.io/en/latest/io/index.html>`_ wherever possible but also fee free to add additional I/O functionality as necessary.

   Test you code internally. We aim to add unittests in the future, feel free to make a start on that.

4. :ref:`Write an example <examples>` showcasing your new function

5. To get your code into the Nighres master follow :ref:`adapt-docs` and :ref:`make-pr`

6. Pat yourself on the back, you have now joined the Nighres developer community!

.. admonition:: A word on dependencies

   One of Python's strengths are the many different packages with specific functionalities. We try to keep the dependencies for Nighres as slim as possible, so please keep that in mind while you are coding. For example, if you need to perform a correlation and would normally use scipy.stats.pearsonr maybe you can get by with numpy.coeff (numpy is already a dependency, scipy isn't). If additional packages are required, they should be pip installable and potentially relevant to other functionality that may be coded in the future.
