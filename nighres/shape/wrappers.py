from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .topology_correction import topology_correction

# TopologyCorrectionInputSpec, TopologyCorrectionOutputSpec, TopologyCorrection
class TopologyCorrectionInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class TopologyCorrectionOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class TopologyCorrection(BaseInterface):
    input_spec  = TopologyCorrectionInputSpec
    output_spec = TopologyCorrectionOutputSpec

    def _run_interface(self, runtime):
        topology_correction(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
