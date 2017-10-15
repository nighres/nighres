.. _python-function:

Adding a new Python function
=============================
We would be happy to have you contribute a python function or module to Nighres, lets get into it!.

0. Before you begin:
Please take a look at the code/modules that already `exist <http://nighres.readthedocs.io/en/latest/index.html>` and are under development. If the functionality that you are looking for exists, your job is done (but maybe you want to enhance the use case...?). If similar functionality is under development or discussion <https://github.com/nighres/nighres/issues>, get in touch with the developer(s) directly and combine your efforts.
Take a moment to familiarise yourself with the documentation and functions, it is likely that some helper functions that you will want to use (e.g., `I/O <http://nighres.readthedocs.io/en/latest/io/index.html>`) have already been coded and tested.

1. Fork, clone, and create/checkout a new branch:
As described here: http://nighres.readthedocs.io/en/latest/developers/setup.html

2. Deciding where to put your code:
If the functionality that you are adding falls within the scope of a pre-existing module, edit/create files within the module. If it is entirely new (e.g., statistics!), create a new module with its own dedicated subdirectory.

3. Coding:
If you are creating a module from scratch, an easy way to start is to copy the __init.py__ and initial import statements from an existing module. 

We are attempting to keep the code approximately `PEP8 compliant <https://www.python.org/dev/peps/pep-0008/>`, so following best practices for labeling of functions that are only used internally (with a leadning "_") and keeping line lengths reasonable will be greatly appreciated. As described here, please keep within the Nighres documentation guidelines: http://nighres.readthedocs.io/en/latest/developers/docs.html - this makes it less painful for everyone to keep track of documentation. Commenting within your code is also good, particularly for individuals who may want to understand and adapt your code later... 

Now actually get in there and code. Use the standard I/O interfaces wherever possible (http://nighres.readthedocs.io/en/latest/io/index.html), but also fee free to add additional I/O functionality within this module as necessary. 

A word about dependencies. As you no doubt know, one of the strengths of Python is that it has many many different packages with specific functionalities (you may have heard, for example, of an amazing package called `Nighres <http://nighres.readthedocs.io/en/latest/>` ;-) ). Our philosophy is to try to keep the dependencies for Nighres as slim as possible, so please keep that in mind while you are coding. For example, if you need to perform a correlation and would normally use scipy.stats.pearsonr (which would necessitate adding scipy to the list of Nighres dependencies) maybe you can get by with numpy.coeff. If additional packages are required, they should be pip installable and (as much as possible) potentially relevant to other functionality that may be coded in the future. Examples include, but are not limited to: scipy, nipy, nilearn, pandas, and statsmodels.

4. Testing:
Test extensively internally before making a pull request. `Write tests <http://nighres.readthedocs.io/en/latest/developers/tests.html>`.

5. Getting your code into the Nighres master:
`Adapt the Nighres documentation <http://nighres.readthedocs.io/en/latest/developers/docs.html>`
`Make a pull request <http://nighres.readthedocs.io/en/latest/developers/pr.html>`

6. Pat yourself on the back, you have now joined the Nighres developer community!
