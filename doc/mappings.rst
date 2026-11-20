.. _mappings:

Mappings
==========

One of the biggest challenges in practical analysis of neuroimaging experiments 
is data registration, both of heterogeneous data within subjects and across
groups of subjets and templates. Coordinate systems and transformations can be
defined in many ways, and almost every image analysis package uses its own 
convention to record and apply transformations...

Nighres is no exception: it uses coordinate mappings (rather than e.g. transformation
matrices and vectr fields) to define transformations, both linear and non-linear.
A coordinate mapping aims to reduce the number of implict modeling conventions: 
it simply records for each voxel of the target image the (X,Y,Z) location of the 
corresponding point in the source image. Thus there is no need to assume anything
about coordinate systems, image resolutions or origins, and everything is expressed
in voxel space. On the other hand, the transformations are not straightforward to
look at, as they are smoothly varying, mostly linear fields.

Coordinate mappings are the basis for all the tools in :func:`nighres.registration`
including when wrapping external registration functions from 
`antspyx <https://github.com/ANTsX/ANTsPy/>`_, translating its own format for 
transformations into coordinate mappings. Some functions only generate mappings,
and you need to use :func:`nighres.registration.apply_coordinate_mappings` to
transform images.

.. todo:: example of 
   using coordinate mappings  with :func:`nighres.registration.apply_coordinate_mappings`
