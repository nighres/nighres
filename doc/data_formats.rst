.. _data-formats:

Data handling and formats
==========================

Nighres represents data internally using `NibabelSpatialImage objects
<http://nipy.org/nibabel/reference/nibabel.spatialimages.html#nibabel.spatialimages.SpatialImage>`_
which are refered to as ``niimg``.

Much of the input and output functionality has been adopted or inspired from
`Nilearn's conventions for data handling
<http://nilearn.github.io/manipulating_images/input_output.html>`_

.. todo:: Explanation why this is useful and little example how it works,
   also mention dictionary outputs
