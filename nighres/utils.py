import os
import warnings
import psutil
from nighres.global_settings import TOPOLOGY_LUT_DIR, MGDM_ATLAS_DIR, DEFAULT_MGDM_ATLAS


def _output_dir_4saving(output_dir=None, rootfile=None):
    if (output_dir is None or output_dir==''):
        if rootfile is None:
            # if nothing is specified, use current working dir
            output_dir = os.getcwd()
        else:
            # if rootfile is specified, use its directory
            output_dir = os.path.dirname(rootfile)
            # if rootfile is in current directory, dirname returns ''
            if (output_dir is None or output_dir==''):
               output_dir = os.getcwd()

   # create directory recursively if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # make sure path ends on seperator
    if not(output_dir[-1] == os.path.sep):
        output_dir += os.path.sep

    # check if there is write access to the directory
    if not os.access(output_dir, os.W_OK | os.X_OK):
        raise ValueError("Cannot write to {0}, please specify a different "
                         "output_dir. (Note that if you don't set output_dir "
                         "explicitly, it will be set to the directory of the "
                         "input file, if applicable, or to the current "
                         "working directory otherwise)".format(output_dir))

    print("\nOutputs will be saved to {0}".format(output_dir))
    return output_dir

## preferred: use given extension (see below)
def _fname_4saving_prev(file_name=None, rootfile=None, suffix=None, ext=None, module='output'):

    # if a file_name is given, use that
    if file_name is None:
        # if a rootfile is given (which is a file_name and not a data object)
        # use its file_name
        #python2 if isinstance(rootfile, basestring):
        if isinstance(rootfile, str):
            file_name = os.path.basename(rootfile)
            #print(("You have not specified a file_name. We will use the "
            #       "name of your input ({0}) as a base name for saving "
            #       "outputs.".format(file_name)))
            # if there is no suffix set trivial one to avoid overriding input
            if suffix is None:
                suffix = 'out'

        # if nothing is given, raise error
        else:
            file_name = module+'.nii.gz'
#            raise ValueError("You have not specified a file_name, and we "
#                             "cannot determine a name from your input, wich "
#                             "is a data object. Please specify a file_name.")

    # avoid empty strings
    if len(file_name) <= 1:
        raise ValueError("Empty string for file_name. Check if your inputs "
                          "exist, or try to specify the file_name "
                          "parameter for saving.".format(file_name))

    # split off extension
    split_name = file_name.split('.')
    # if there was no dot in the file_name set nii.gz as extension (not
    # foolproof, if the name is e.g. 'hello.bello' without
    # extension it will think bello is the extension)
    if len(split_name) == 1:
        base = split_name[0]
        if ext is None: ext = 'nii.gz'
    else:
        # pop file extension
        if ext is None:
            ext = split_name.pop(-1)
            # file extension could have two parts if compressed
            if ext == 'gz':
                ext = split_name.pop(-1)+'.gz'
        # now that the extension has been popped out of the list
        # what's left is the basename, put back together
        base = split_name.pop(0)
        while split_name:
            base += '.'+split_name.pop(0)

    # insert suffix if given
    if suffix is not None:
        fullname = base + '_' + suffix + '.' + ext
    else:
        fullname = base + '.' + ext

    return fullname


def _fname_4saving(file_name=None, rootfile=None, suffix=None, ext=None, module='output'):

    # default extension if not given
    file_ext = 'nii.gz'
    # if a file_name is given, use that
    if file_name is None:
        # if a rootfile is given (which is a file_name and not a data object)
        # use its file_name
        #python2 if isinstance(rootfile, basestring):
        if isinstance(rootfile, str):
            file_name = os.path.basename(rootfile)
            #print(("You have not specified a file_name. We will use the "
            #       "name of your input ({0}) as a base name for saving "
            #       "outputs.".format(file_name)))
            # if there is no suffix set trivial one to avoid overriding input
            if suffix is None:
                suffix = 'out'

        # if nothing is given, raise error
        else:
            file_name = module+'.nii.gz'
#            raise ValueError("You have not specified a file_name, and we "
#                             "cannot determine a name from your input, wich "
#                             "is a data object. Please specify a file_name.")

    # avoid empty strings
    if len(file_name) <= 1:
        raise ValueError("Empty string for file_name. Check if your inputs "
                          "exist, or try to specify the file_name "
                          "parameter for saving.".format(file_name))

    # split off extension
    split_name = file_name.split('.')
    # if there was no dot in the file_name set nii.gz as extension (not
    # foolproof, if the name is e.g. 'hello.bello' without
    # extension it will think bello is the extension)
    if len(split_name) == 1:
        base = split_name[0]
    else:
        # pop file extension
        file_ext = split_name.pop(-1)
        # file extension could have two parts if compressed
        if file_ext == 'gz':
            file_ext = split_name.pop(-1)+'.gz'
        # now that the extension has been popped out of the list
        # what's left is the basename, put back together
        base = split_name.pop(0)
        while split_name:
            base += '.'+split_name.pop(0)

        # Check if extension is given, otherwise use from file name
        if ext is None:
            ext = file_ext

    # If there was no extension given and the file name didn't have extension
    # use nifti
    if ext is None:
        ext = 'nii.gz'

    # insert suffix if given
    if suffix is not None:
        fullname = base + '_' + suffix + '.' + ext
    else:
        fullname = base + '.' + ext

    return fullname


def _check_topology_lut_dir(topology_lut_dir):

    if topology_lut_dir is None:
        topology_lut_dir = TOPOLOGY_LUT_DIR
    else:
        # check if dir exists
        if not os.path.isdir(topology_lut_dir):
            raise ValueError('The topology_lut_dir you have specified ({0}) '
                             'does not exist'.format(topology_lut_dir))
    # make sure there is a  trailing slash
    topology_lut_dir = os.path.join(topology_lut_dir, '')

    return topology_lut_dir


def _check_mgdm_atlas_file(atlas_file):

    if atlas_file is None:
        atlas_file = DEFAULT_MGDM_ATLAS
    else:
        # check if file exists, if not try search atlas in default atlas dir
        if not os.path.isfile(atlas_file):
            if not os.path.isfile(os.path.join(MGDM_ATLAS_DIR, atlas_file)):
                raise ValueError('The atlas_file you have specified ({0}) '
                                 'does not exist'.format(atlas_file))
            else:
                atlas_file = os.path.join(MGDM_ATLAS_DIR, atlas_file)

    return atlas_file


def _check_available_memory():

    init_memory = str(int(round(0.25*psutil.virtual_memory()[1])))
    max_memory = str(int(round(0.95*psutil.virtual_memory()[1])))

    return {"init": init_memory, "max": max_memory}
