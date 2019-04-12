from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .recursive_ridge_diffusion import recursive_ridge_diffusion

# RecursiveRidgeDiffusionInputSpec, RecursiveRidgeDiffusionOutputSpec, RecursiveRidgeDiffusion
class RecursiveRidgeDiffusionInputSpec(BaseInterfaceInputSpec):

    input_image = File(exists=True, desc='specify an input image', mandatory=True)
    ridge_intensities = traits.Str(argstr='%s', desc='ridge intensities', mandatory=True)
    ridge_filter = traits.Str(argstr='%s', desc='ridge filter', mandatory=True)
    surface_levelset = File(exists=True, desc='specify a distance image')
    orientation = traits.Str(argstr='%s', desc='orientation', mandatory=True)
    ang_factor = traits.Float(desc='angular factor',mandatory=True)
    loc_prior = File(exists=True, desc='specify a location prior image')
    min_scale = traits.Int(desc='minimum scale', mandatory=True)
    max_scale = traits.Int(desc='maximum scale', mandatory=True)
    propagation_model = traits.Str(argstr='%s', desc='propagation model', mandatory=True)
    diffusion_factor = traits.Float(desc='diffusion factor',mandatory=True)
    similarity_scale = traits.Float(desc='similarity scale',mandatory=True)
    neighborhood_size = traits.Int(desc='neighborhood size', mandatory=True)
    max_iter = traits.Int(desc='maximum of iterations', mandatory=True)
    max_diff = traits.Float(desc='maximum difference',mandatory=True)
    save_data = traits.Bool(True, desc='Save output data to file', usedefault=True)
    output_dir = traits.Str(argstr='%s', desc='output directory', mandatory=True)
    #file_name=None)


class RecursiveRidgeDiffusionOutputSpec(TraitedSpec):

    ridge_pv = File(exists=True, desc="")
    filter = File(exists=True, desc="")
    proba = File(exists=True, desc="")
    propagation = File(exists=True, desc="")
    scale = File(exists=True, desc="")
    ridge_direction = File(exists=True, desc="")
    correction = File(exists=True, desc="")
    ridge_size = File(exists=True, desc="")


class RecursiveRidgeDiffusion(BaseInterface):
    input_spec = RecursiveRidgeDiffusionInputSpec
    output_spec = RecursiveRidgeDiffusionOutputSpec

    def _run_interface(self, runtime):

        recursive_ridge_diffusion(input_image = self.inputs.input_image,
                                  ridge_intensities = self.inputs.ridge_intensities,
                                  ridge_filter = self.inputs.ridge_filter,
                                  surface_levelset = self.inputs.surface_levelset,
                                  orientation = self.inputs.orientation,
                                  ang_factor = self.inputs.ang_factor,
                                  loc_prior = self.inputs.loc_prior,
                                  min_scale = self.inputs.min_scale,
                                  max_scale = self.inputs.max_scale,
                                  propagation_model = self.inputs.propagation_model,
                                  diffusion_factor = self.inputs.diffusion_factor,
                                  similarity_scale = self.inputs.similarity_scale,
                                  neighborhood_size = self.inputs.neighborhood_size,
                                  max_iter = self.inputs.max_iter,
                                  max_diff = self.inputs.max_diff,
                                  save_data = self.inputs.save_data,
                                  output_dir = self.inputs.output_dir)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.input_image
        path, base, ext = split_filename(fname)
        outputs["ridge_pv"] = os.path.abspath(base + '_rrd_pv.nii.gz')
        outputs["filter"] = os.path.abspath(base + '_rrd_filter.nii.gz')
        outputs["proba"] = os.path.abspath(base + '_rrd_proba.nii.gz')
        outputs["propagation"] = os.path.abspath(base + '_rrd_propag.nii.gz')
        outputs["scale"] = os.path.abspath(base + '_rrd_scale.nii.gz')
        outputs["ridge_direction"] = os.path.abspath(base + '_rrd_dir.nii.gz')
        outputs["correction"] = os.path.abspath(base + '_rrd_correct.nii.gz')
        outputs["ridge_size"] = os.path.abspath(base + '_rrd_size.nii.gz')
        return outputs
