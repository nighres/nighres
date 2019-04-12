from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .mgdm_cells import mgdm_cells

# MGDMCellsInputSpec, MGDMCellsOutputSpec, MGDMCells
class MGDMCellsInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class MGDMCellsOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class MGDMCells(BaseInterface):
    input_spec  = MGDMCellsInputSpec
    output_spec = MGDMCellsOutputSpec

    def _run_interface(self, runtime):
        mgdm_cells(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
