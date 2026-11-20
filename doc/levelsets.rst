.. _levelsets:

Levelsets
==========

Most of Nighres tools involving surfaces use a 'levelset' representation. Instead of
using an explicit 3D mesh of nodes and triangles in continuous space, a levelset
surface provides an implicit representation in voxel space by recording the distance
of each voxel to the surface of interest. By construction, such surface fully 
encloses an object, and the inside and outside are distinguished by making the 
inside distances negative.

Levelset representations come with all sorts of tools for surface evolution and
local geometry, and can always be translated into triangular meshes using a 
marching cubes algorithm. Their resolution is directly linked to the image
resolution, and most issues related to deforming 3D meshes are avoided by the
use of the representation.

In Nighres, levelsets are used for cortical reconstruction, laminar analysis,
shape processing, and translation functions are available to turn them into 3D
meshes or probability maps. Levelset images themselves are not very intuitive 
to look at as they sho a smoothly varying gradient field: their shape only gets
apparent when thresholding them at zero.


Levelset tools in Nighres
-------------------------

Translation utilities:

:func:`nighres.surface.probability_to_levelset`
:func:`nighres.surface.levelset_to_probability`

:func:`nighres.surface.levelset_to_mesh`
:func:`nighres.surface.mesh_to_levelset`


Shape analysis:

:func:`nighres.surface.levelset_curvature`
:func:`nighres.shape.levelset_thickness`
:func:`nighres.shape.simple_skeleton`
:func:`nighres.shape.topology_correction`
:func:`nighres.shape.levelset_fusion`


Cortical and laminar analysis:

:func:`nighres.brain.mgdm_segmentation`
:func:`nighres.cortex.cruise_cortex_extraction`
:func:`nighres.laminar.volumetric_layering`
:func:`nighres.laminar.profile_sampling`
:func:`nighres.laminar.laminar_iterative_smoothing`


.. todo:: example of creating a
   levelset using :func:`nighres.surface.probability_to_levelset`
