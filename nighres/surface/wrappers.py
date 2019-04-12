from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .probability_to_levelset import probability_to_levelset

# ProbabilityToLevelsetInputSpec, ProbabilityToLevelsetOutputSpec, ProbabilityToLevelset

class ProbabilityToLevelsetInputSpec(BaseInterfaceInputSpec):

    probability_image = File(exists=True, desc='specify a probability image', mandatory=True)
    save_data = traits.Bool(True, desc='Save output data to file', usedefault=True)
    output_dir = traits.Str(argstr='%s', desc='output directory', mandatory=True)
    #file_name=None)


class ProbabilityToLevelsetOutputSpec(TraitedSpec):

    levelset = File(exists=True, desc="")


class ProbabilityToLevelset(BaseInterface):
    input_spec = ProbabilityToLevelsetInputSpec
    output_spec = ProbabilityToLevelsetOutputSpec

    def _run_interface(self, runtime):

        probability_to_levelset(probability_image = self.inputs.probability_image,
                                save_data = self.inputs.save_data,
                                output_dir = self.inputs.output_dir)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.probability_image
        path, base, ext = split_filename(fname)
        outputs["levelset"] = os.path.abspath(base + '_levelset.nii.gz')
        return outputs
