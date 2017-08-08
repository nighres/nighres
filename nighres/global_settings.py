import os

__dir__ = os.path.abspath(os.path.dirname(__file__))

ATLAS_DIR = os.path.join(__dir__, 'atlases')
TOPOLOGY_LUT_DIR = os.path.join(ATLAS_DIR, 'topology_lut')
DEFAULT_ATLAS = os.path.join(ATLAS_DIR, 'brain-segmentation-prior3.0',
                             'brain-atlas-3.0.3.txt')
