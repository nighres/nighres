import os
import warnings
from global_settings import TOPOLOGY_LUT_DIR, ATLAS_DIR, DEFAULT_ATLAS


def _output_dir_4saving(output_dir=None, rootfile=None):
    if (output_dir is None or output_dir==''):
        if rootfile is None:
            # if nothing is specified, use current working dir
            output_dir = os.getcwd()
        else:
            # if rootfile is specified, use it's directory
            output_dir = os.path.dirname(rootfile)

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
                         "working directory otherwise)").format(output_dir)

    print("\nOutputs will be saved to {0}").format(output_dir)
    return output_dir


def _fname_4saving(file_name=None, rootfile=None, suffix=None):

    # if a file_name is given, use that
    if file_name is None:
        # if a rootfile is given (which is a file_name and not a data object)
        # use its file_name
        if isinstance(rootfile, basestring):
            file_name = os.path.basename(rootfile)
            #print(("You have not specified a file_name. We will use the "
            #       "name of your input ({0}) as a base name for saving "
            #       "outputs.").format(file_name))
            # if there is no suffix set trivial one to avoid overriding input
            if suffix is None:
                suffix = 'out'

        # if nothing is given, raise error
        else:
            raise ValueError("You have not specified a file_name, and we "
                             "cannot determine a name from your input, wich "
                             "is a data object. Please specify a file_name.")

    # avoid empty strings
    if len(file_name) <= 1:
        raise ValueError(("Empty string for file_name. Check if your inputs "
                          "exist, or try to specify the file_name "
                          "parameter for saving.").format(file_name))

    # split off extension
    split_name = file_name.split('.')
    # if there was no dot in the file_name set nii.gz as extension (not
    # foolproof, if the name is e.g. 'hello.bello' without
    # extension it will think bello is the extension)
    if len(split_name) == 1:
        base = split_name[0]
        ext = 'nii.gz'
    else:
        # pop file extension
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


def _check_atlas_file(atlas_file):

    if atlas_file is None:
        atlas_file = DEFAULT_ATLAS
    else:
        # check if file exists, if not try search atlas in default atlas dir
        if not os.path.isfile(atlas_file):
            if not os.path.isfile(os.path.join(ATLAS_DIR, atlas_file)):
                raise ValueError('The atlas_file you have specified ({0}) '
                                 'does not exist'.format(atlas_file))
            else:
                atlas_file = os.path.join(ATLAS_DIR, atlas_file)

    return atlas_file
