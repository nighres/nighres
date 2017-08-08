import os
import warnings
import urllib


def _output_dir_4saving(output_dir=None, rootfile=None):
    if output_dir is None:
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

    print("\n Outputs will be saved to {0}").format(output_dir)
    return output_dir


def _fname_4saving(rootfile=None, suffix=None,
                   base_name=None, extension=None):
    # if both base_name and extension are given, jump right to inserting suffix
    if (base_name is None or extension is None):
        # else, if a rootfile is given, find base_name and extension
        if isinstance(rootfile, basestring):
            # avoid empty strings
            if len(rootfile) <= 1:
                raise ValueError("When trying to determine the file name for "
                                 "saving from the input {0}, the input file "
                                 "name seems empty. Check if your inputs "
                                 "exist, or try to specify the base_name "
                                 "parameter for saving.").format(rootfile)
            split_root = os.path.basename(rootfile).split('.')
            # if there was only one dot in the filename, it is good to go
            if len(split_root) == 2:
                base = split_root[0]
                ext = split_root[1]
            else:
                # pop file extension
                ext = split_root.pop(-1)
                # file extension could have two parts if compressed
                if ext == 'gz':
                    ext = split_root.pop(-1)+'.gz'
                # now that the extension has been popped out of the list
                # what's left is the basename, put back together
                base = split_root.pop(0)
                while split_root:
                    base += '.'+split_root.pop(0)
            # use rootfile parts only for what's missing
            if not base_name:
                base_name = base
            if not extension:
                extension = ext
        # if the input is not a filename but a data object both base_name
        # and extension should be given, raise warning and make surrogate
        else:
            if base_name is None:
                base_name = 'output'
            if extension is None:
                extension = 'nii.gz'

            warnings.warn(("If passing a data object as input, you should "
                           "specify a base_name AND extension for saving. "
                           "Saving to %s.%s (plus suffixes) for now")
                          % (base_name, extension))

    # insert suffix if given
    if suffix is not None:
        base_name += '_' + suffix

    # putting it all together
    fname = base_name + '.' + extension

    return fname


def _download_from_url(url, filename, overwrite_file=False):

    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if os.path.isfile(filename) and overwrite_file is False:
        print("The file {0} exists and overwrite_file was set to False. "
              "Not downloading.").format(filename)
    else:
        print("Downloading to {0}").format(filename)
        urllib.urlretrieve(url, filename)
