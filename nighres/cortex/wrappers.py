from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .cruise_cortex_extraction import cruise_cortex_extraction

# CruiseCortexExtractionInputSpec, CruiseCortexExtractionOutputSpec, CruiseCortexExtraction
class CruiseCortexExtractionInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class CruiseCortexExtractionOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class CruiseCortexExtraction(BaseInterface):
    input_spec  = CruiseCortexExtractionInputSpec
    output_spec = CruiseCortexExtractionOutputSpec

    def _run_interface(self, runtime):
        cruise_cortex_extraction( 
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
