from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

from .extract_brain_region import extract_brain_region
from .filter_stacking import filter_stacking
from .mgdm_segmentation import mgdm_segmentation
from .mp2rage_dura_estimation import mp2rage_dura_estimation
from .mp2rage_skullstripping import mp2rage_skullstripping

# ExtractBrainRegionInputSpec, ExtractBrainRegionOutputSpec, ExtractBrainRegion
class ExtractBrainRegionInputSpec(BaseInterfaceInputSpec):
    segmentation        = File(exists=True, desc='description', mandatory=True),
    levelset_boundary   = File(exists=True, desc='description', mandatory=True),
    maximum_membership  = File(exists=True, desc='description', mandatory=True),
    maximum_label       = File(exists=True, desc='description'),
    extracted_region    = File(exists=True, desc='description'),
    # atlas_file          = None,

    normalize_probabilities     = traits.Bool(False, desc='description', usedefault=True),
    estimate_tissue_densities   = traits.Bool(False, desc='description', usedefault=True),
    # partial_volume_distance     = 1.0,
    save_data                   = traits.Bool(False, desc='description', usedefault=True),
    overwrite                   = traits.Bool(False, desc='description', usedefault=True),
    # output_dir                  = None,
    # file_name                   = None


class ExtractBrainRegionOutputSpec(TraitedSpec):
    region_mask         = File(exists=True, desc="description")
    inside_mask         = File(exists=True, desc="description")
    background_mask     = File(exists=True, desc="description")
    region_proba        = File(exists=True, desc="description")
    inside_proba        = File(exists=True, desc="description")
    background_proba    = File(exists=True, desc="description")
    region_lvl          = File(exists=True, desc="description")
    inside_lvl          = File(exists=True, desc="description")
    background_lvl      = File(exists=True, desc="description")


class ExtractBrainRegion(BaseInterface):
    input_spec  = ExtractBrainRegionInputSpec
    output_spec = ExtractBrainRegionOutputSpec

    def _run_interface(self, runtime):
        extract_brain_region( segmentation              = self.inputs.segmentation,
                              levelset_boundary         = self.inputs.levelset_boundary,
                              maximum_membership        = self.inputs.maximum_membership,
                              maximum_label             = self.inputs.maximum_label,
                              extracted_region          = self.inputs.extracted_region,
                              normalize_probabilities   = self.inputs.normalize_probabilities,
                              estimate_tissue_densities = self.inputs.estimate_tissue_densities,
                              save_data                 = self.inputs.save_data,
                              overwrite                 = self.inputs.overwrite
                            )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)

        # #TODO, update these file names
        outputs["region_mask"]      = os.path.abspath(base + '_mgdm_seg.nii.gz')
        outputs["inside_mask"]      = os.path.abspath(base + '_mgdm_lbls.nii.gz')
        outputs["background_mask"]  = os.path.abspath(base + '_mgdm_mems.nii.gz')
        outputs["region_proba"]     = os.path.abspath(base + '_mgdm_dist.nii.gz')
        outputs["inside_proba"]     = os.path.abspath(base + '_mgdm_dist.nii.gz')
        outputs["background_proba"] = os.path.abspath(base + '_mgdm_dist.nii.gz')
        outputs["region_lvl"]       = os.path.abspath(base + '_mgdm_dist.nii.gz')
        outputs["inside_lvl"]       = os.path.abspath(base + '_mgdm_dist.nii.gz')
        outputs["background_lvl"]   = os.path.abspath(base + '_mgdm_dist.nii.gz')
        return outputs


# MGDMSegmentationInputSpec, MGDMSegmentationOutputSpec, MGDMSegmentation
class MGDMSegmentationInputSpec(BaseInterfaceInputSpec):
    contrast_image1 = File(exists=True, desc='specify a first contrast image', mandatory=True)
    contrast_type1  = traits.Str(argstr='%s', desc='type of the image number one', mandatory=True)
    contrast_image2 = File(exists=True, argstr="%s", desc='specify a second contrast image (optionnal)')
    contrast_type2  = traits.Str(argstr='%s', desc='type of the image number two')
    contrast_image3 = File(exists=True, argstr="%s", desc='specify a third contrast image (optionnal)')
    contrast_type3  = traits.Str(argstr='%s', desc='type of the image number three')
    #contrast_image4 = File(exists=True, argstr="%s", desc='specify a fourth contrast image (optionnal)')
    #contrast_type4  = traits.Str(argstr='%s', desc='type of the image number four')

    #n_steps
    #max_iterations
    #topology='wcs'

    atlas_file = File(exists=True, desc='Path to plain text atlas file', mandatory=True)
    #topology_lut_dir=None,
    #adjust_intensity_priors=False,
    #compute_posterior=False,
    #diffuse_probabilities=False,
    save_data = traits.Bool(True, desc='Save output data to file', usedefault=True)
    output_dir = traits.Str(argstr='%s', desc='output directory', mandatory=True)
    #file_name=None


class MGDMSegmentationOutputSpec(TraitedSpec):
    segmentation = File(exists=True, desc="Hard brain segmentation with topological constraints (if chosen)")
    labels = File(exists=True, desc="Maximum tissue probability labels")
    memberships = File(exists=True, desc="""Maximum tissue probability values, 4D image where the first dimension shows each voxel's highest probability to  belong to a specific tissue,
                                            the second dimension shows the second highest probability to belong to another tissue etc.""")
    distance = File(exists=True, desc="Minimum distance to a segmentation boundary")


class MGDMSegmentation(BaseInterface):
    input_spec = MGDMSegmentationInputSpec
    output_spec = MGDMSegmentationOutputSpec

    def _run_interface(self, runtime):

        mgdm_segmentation(contrast_image1 = self.inputs.contrast_image1,
                          contrast_type1 = self.inputs.contrast_type1,
                          contrast_image2 = self.inputs.contrast_image2,
                          contrast_type2 = self.inputs.contrast_type2,
                          contrast_image3 = self.inputs.contrast_image3,
                          contrast_type3 = self.inputs.contrast_type3,
                          atlas_file = self.inputs.atlas_file,
                          save_data = self.inputs.save_data,
                          output_dir = self.inputs.output_dir)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.contrast_image1
        path, base, ext = split_filename(fname)
        outputs["segmentation"] = os.path.abspath(base + '_mgdm_seg.nii.gz')
        outputs["labels"] = os.path.abspath(base + '_mgdm_lbls.nii.gz')
        outputs["memberships"] = os.path.abspath(base + '_mgdm_mems.nii.gz')
        outputs["distance"] = os.path.abspath(base + '_mgdm_dist.nii.gz')
        return outputs

#TODO Add these functions as wrappers

# FilterStackingInputSpec, FilterStackingOutputSpec, FilterStacking
# MP2RAGEDuraInputSpec, MP2RAGEDuraOutputSpec, MP2RAGEDuraEstimation
# MP2RAGESkullStrippingInputSpec, MP2RAGESkullStrippingOutputSpec, MP2RAGESkullStripping
