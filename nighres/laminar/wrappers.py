from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .laminar_iterative_smoothing import laminar_iterative_smoothing

# LaminarIterativeSmoothingInputSpec, LaminarIterativeSmoothingOutputSpec, LaminarIterativeSmoothing
class LaminarIterativeSmoothingInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class LaminarIterativeSmoothingOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class LaminarIterativeSmoothing(BaseInterface):
    input_spec  = LaminarIterativeSmoothingInputSpec
    output_spec = LaminarIterativeSmoothingOutputSpec

    def _run_interface(self, runtime):
        laminar_iterative_smoothing(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
