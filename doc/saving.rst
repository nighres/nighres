Saving outputs
==============

Each Nighres processing interface allows you to save the outputs by setting the ``save_data`` parameter to ``True``. If this parameter is not specified, it defaults to ``False`` and the data is returned as a data object (see :ref:`data-formats`) but not saved to disk.

If ``save_data`` is set to True, Nighres applies the following logic:

**Output directory**

1. If ``output_dir`` is specified, the data is saved there. In case ``output_dir`` doesn't exist, it is created
2. If ``output_dir`` is not specified, Nighres tries to use the location of an input file as the location for saving. This only works if the input is a file name and not a data object
3. Otherwise, the data is saved in the current working directory

**File names**

1. If ``file_name`` is specified, this name is used as a base to create the output names. A suffix is added to each output (you can see in the docstrings which suffix refers to which output). The extension of ``file_name`` specifies the format in which the output will be saved. If ``file_name`` has no extension, Nighres defaults to *nii.gz*
2. If ``file_name`` is not specified, Nighres tries to use the name of an input file as a base name for saving. This only works if the input is indeed a file name and not a data object
