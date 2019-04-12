from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .background_estimation import background_estimation

# BackgroundEstimationInputSpec, BackgroundEstimationOutputSpec, BackgroundEstimation
class BackgroundEstimationInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class BackgroundEstimationOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class BackgroundEstimation(BaseInterface):
    input_spec  = BackgroundEstimationInputSpec
    output_spec = BackgroundEstimationOutputSpec

    def _run_interface(self, runtime):
        background_estimation(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
