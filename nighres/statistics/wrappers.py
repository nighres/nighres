from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .segmentation_statistics import segmentation_statistics

# SegmentationStatisticsInputSpec, SegmentationStatisticsOutputSpec, SegmentationStatistics
class SegmentationStatisticsInputSpec(BaseInterfaceInputSpec):
    output_dir                  = None,


class SegmentationStatisticsOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")


class SegmentationStatistics(BaseInterface):
    input_spec  = SegmentationStatisticsInputSpec
    output_spec = SegmentationStatisticsOutputSpec

    def _run_interface(self, runtime):
        segmentation_statistics(
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        return outputs
