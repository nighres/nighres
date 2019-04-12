from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .io_mesh import load_mesh

# LoadMeshInputSpec, LoadMeshOutputSpec, LoadMesh
class LoadMeshInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class LoadMeshOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class LoadMesh(BaseInterface):
    input_spec  = LoadMeshInputSpec
    output_spec = LoadMeshOutputSpec

    def _run_interface(self, runtime):
        load_mesh(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
