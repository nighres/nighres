.. _set-up:

Setting up
==========

**1. Fork and clone the** `Nighres github repository <https://github.com/nighres/nighres>`_

   You can find a good description `here <https://help.github.com/articles/fork-a-repo/>`_. Make sure that you set up your local clone to track both your fork (typically as *origin*) as well as the original Nighres repo (typically as *upstream*).

**2. Make a new branch to work on**

   ``git checkout -b <branch_name>``

   This is important. When you make a pull request later, we will likely ask you to `rebase <https://help.github.com/articles/about-git-rebase/>`_ because the Nighres master might have moved on since you forked it. You need a clean master branch for that.

   Pick a descriptive branch name. For example, when you fix a bug in the function load_volume a good name is ``fix/load_volume``. When you write a new interface called laminar_connectivity (that would be cool!) a good name is ``enh/laminar_connectivity``.

   Make one branch for each new feature or fix. That way independent changes can be handled in different :ref:`pull requests <make-pr>` later.

**3. Install in editable mode (optional)**

   ``pip install -e <path_to_nighres_directory>``

   This way, when you import Nighres functions in Python, they will always come from the current version of the code in your Nighres directory. This is convenient for testing your code while developing.

   (Alternatively, you can stay inside your Nighres directory and import functions directly from there without installing at all)

**4. Let the coding begin!**

   If you want to work on an existing function, you can most likely just make your changes, check if the :ref:`Examples <examples>` still run and the move on to :ref:`adapt-docs` and :ref:`make-pr`.

   If you want to add new functions be sure to check out our intructions on :ref:`wrap-cbstools` or :ref:`python-function`

.. important:: Please adhere to `PEP8 style conventions
   <https://www.python.org/dev/peps/pep-0008/>`_. This is easiest by using a Python linter which is available in most editors.
