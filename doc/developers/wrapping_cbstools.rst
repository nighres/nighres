.. _wrap-cbstools:

Wrapping an existing CBS Tools class
====================================
You want to wrap an existing CBS Tools module into Nighres? Great!

How does it work?
-----------------
The CBS Tools modules have been first created for JIST. Most of them are
however not directly dependent on the JIST and MIPAV libraries, which is why
we can turn them into a JCC-compatible, lightweight library.

JCC requires that all data is passed as numbers, strings, and 1D arrays, so this
is what the CBS Tools modules need to manipulate as input and output, and also
why we created a second python layer for structured data handling. Once a
CBS Tools module has been formatted to the adequate i/o structure, JCC wraps
it and you can call it directly from python.

Standard procedures
-------------------
Each JCC-wrapped CBS Tools module has three important functions: .setSomeParameter(),
.getSomeResult(), and .execute(). You need to set the inputs with meaningful values,
execute the module, and retrieve the results. Note that default parameters are
always assumed when possible, but a good wrapper should expose all parameters.

General wrapper content
-----------------------

**1 Start the Java Virtual Machine (JVM)**

    ``cbstools.initVM(initialheap='6000m', maxheap='6000m')``
    
    Note that the initial and maximum amount of allocated memory may be 
    adjusted or set up as a global parameter.
     
**2 Create an instance of the module**

    ``mymodule = cbstools.SomeCoolModule()``
    
**3 Set all the parameters and data arrays**

    ``my_module.setThisImportantParameter(some_value)``
    ``my_module.setInputImage(cbstools.JArray('float')((my_image_data.flatten('F')).astype(float))``   

**4 Run the module**

    ``my_module.execute()``
    
**5 Retrieve the outputs**

    ``my_result_data = np.reshape(np.array(my_module.getCoolResultImage(), dtype=np.float32), dimensions, 'F')``

    Note that because you are passing simple 1D arrays, you need to keep a record
    of image dimensions, resolutions, headers, etc.
    
Convenience python layer
------------------------
We strongly advise (at least for inclusion into Nighres) to handle the inputs
and outputs of python structures as we do, i.e. loading images from files or
passing them as Nifti1Image, and allowing to save all outputs automatically.

We also adhere to a somewhat strict formatting of the outputs with two suffixes, 
first for the module name, and second for the specific output, e.g. ''_mgdm_seg''
for the segmentation obtained from the mgdm\_brain\_segmentation module.


What if my favorite CBS Tools module is not ready to wrap?
----------------------------------------------------------
Because preparing modules for wrapping requires some reformatting, only a few
modules so far are ready. You can accelerate the process in two ways: 1) send
a request to `the CBS Tools developers <https://www.github.com/piloubazin/>`_ 
or 2) write a reformatted module for CBS Tools yourself and make a pull request.
All the currently wrapped CBS Tools modules follow the same formatting procedure,
so transforming a JIST module definition into a core module (which then can be
called either from JIST or from Nighres) should be reasonably easy in most cases.

