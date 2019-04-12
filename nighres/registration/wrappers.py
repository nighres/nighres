from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .simple_align import simple_align

# SimpleAlignInputSpec, SimpleAlignOutputSpec, SimpleAlign
class SimpleAlignInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class SimpleAlignOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class SimpleAlign(BaseInterface):
    input_spec  = SimpleAlignInputSpec
    output_spec = SimpleAlignOutputSpec

    def _run_interface(self, runtime):
        simple_align(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
