.. _adapt-docs:

Adapting the docs
=================

Nighres uses `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ which makes it easy to automatically pull the docstrings of functions into the online documentation. This means once a function has a good docstring, the work is essentially done. It also means that writing great docstrings is important.

If you changed an existing function
-----------------------------------
Make sure to check if your changes also require changes in the docstring. Adapting an existing docstring is usually straightforward. If you are curious how your changes will look in the online docs, you can try out :ref:`build-docs`.

.. _newfunc-docs:

If you added a new function
-----------------------------
Make sure to write a comprehensive docstring. We use `NumPy/SciPy docstring <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_ conventions. But the easiest is to just copy another function's docstring and adapt. Then you need to add a few things to the ``nighres/doc`` directory:

1. Go to the subdirectory which refers to the submodule you added a function to, e.g. ``nighres/doc/laminar/``.
2. Create a new file named after your new function, e.g. *laminar_connectivity.rst*
3. Add the name of this file in the *index.rst* file of the submodule
4. Copy the content of another function's rst file inside of your new file
5. Adapt the title and the two mentions of the function name in your new file to match your new function
6. Submit the changes to the docs along with your PR

Again, you can check how the changes you made will look in the online documentation by :ref:`build-docs`.

If you added a new submodule
-----------------------------
This is going to be a rare case. But if indeed added a new submodule in your PR, say its called *nighres.fancypants*:

1. In the doc directory, add a new subdirectory for your new module, e.g. ``nighres/doc/fancypants``
2. In that subdirectory make a file called *index.rst*. Best to just copy the index.rst from another submodule. Then adapt the title in the file to the name of your new submodule, and remove all function names in the toctree.
3. Follow steps 3 to 5 from :ref:`newfunc-docs` to add all functions of your new submodule
4. Add your submodule in the *index.rst* file in the ``nighres/doc`` main directory under the "Modules and Functions" toctree. In our case we would add the line *fancypants/index*
5. Submit the changes to the docs along with your PR

More than docstrings
--------------------
You can also make changes to parts of the documentation that are not function docstrings. For example to help improving this guide!

The easiest way is to browse to the page in the `online documentation <http://nighres.readthedocs.io/en/latest/>`_ that you want to change and click on **Edit on Github** to find the location of the file you have to change. For example, if you want to make a change to this page, the link would send you to https://github.com/nighres/nighres/blob/master/doc/developers/docs.rst, which tells you that in your local copy of the repo you should edit the file *nighres/doc/developers/docs.rst*.

Sphinx uses reStructuredText (reST) syntax. For formatting take a look at this `Sphinx Cheatsheet <http://matplotlib.org/sampledoc/cheatsheet.html>`_.

You can also add whole new parts to the online documentation. This might require learning a bit more about `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. You can also open an `issue <https://github.com/nighres/nighres/issues>`_ and get help.

When you are done, you can check your changes by :ref:`build-docs`.

.. _build-docs:

Building the docs locally
--------------------------
If you make changes to the documentation, they will only appear online at http://nighres.readthedocs.io/en/latest/ once your PR has been merged and a new release was made. You can, however, build the docs locally for a preview:

1. Make sure to have all :ref:`additional dependencies <add-deps>` installed that you need to build the documentation
2. Navigate into your local copy of ``nighres/doc``
3. Type ``make clean`` and then ``make html``
4. Open an Internet browser and point it to the local html pages that were just built for you, e.g. file:///home/nighres/doc/_build/html/index.html

You can make changes and repeat this process until you are satisfied.
