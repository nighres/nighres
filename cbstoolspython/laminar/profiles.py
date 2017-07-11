def profile_sampling(boundary_img, intensity_img,
                     save_data=True, base_name=None):

    '''
    Sampling data on multiple intracortical layers.

        Parameters
        -----------
        boundary_img : Levelset representations of different intracortical
            layers in a 4D image (4th dimensions representing the layers).
            Can be created from GM and WM leveset with the "layering" function.
            Can be path to a Nifti file or Nibabel image object.
        intensity_img : Image from which data should be sampled. Can be path to
            a Nifti file or Nibabel image object.
        save_data : Whether the output profile image should be saved
            (default is 'True').
        base_name : If save_data is set to True, this parameter can be used to
            specify where the output should be saved. Thus can be the path to a
            directory or a full filename. The suffix 'profiles' will be added
            to filename. If None (default), the output will be saved to the
            current directory.

        Returns
        -------
        Nibabel image object (4D), where the 4th dimension represents the
        different cortical surfaces, i.e. the profile for each voxel in the
        3D space.
    '''

    # load the data as well as filenames and headers for saving later
    boundary_img = load_volume(boundary_img)
    boundary_data = boundary_img.get_data()
    hdr = boundary_img.get_header()
    aff = boundary_img.get_affine()

    intensity_data = load_volume(intensity_img).get_data()

    try:
        cbstoolsjcc.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    sampler = cbstoolsjcc.LaminarProfileSampling()
    sampler.setIntensityImage(cbstoolsjcc.JArray('float')((intensity_data.flatten('F')).astype(float)))
    sampler.setProfileSurfaceImage(cbstoolsjcc.JArray('float')((boundary_data.flatten('F')).astype(float)))
    zooms = [x.item() for x in hdr.get_zooms()]
    sampler.setResolutions(zooms[0], zooms[1], zooms[2])
    sampler.setDimensions(boundary_data.shape)
    sampler.execute()

    profile_data = np.reshape(np.array(sampler.getProfileMappedIntensityImage(),
                              dtype=np.float32), boundary_data.shape,'F')

    profile_img = nb.Nifti1Image(profile_data, aff, hdr)

    if save_data:
        if base_name:
            base_name += '_'
        else:
            if not isinstance(intensity_img, basestring):
                base_name = os.getcwd() + '/'
                print "saving to %s" % base_name
            else:
                dir_name = os.path.dirname(intensity_img)
                base_name = os.path.basename(intensity_img)
                base_name = os.path.join(dir_name,
                                         base_name[:base_name.find('.')]) + '_'
        save_volume(base_name+'profiles.nii.gz', profile_img)

    return profile_img
