import nibabel as nb
import numpy as np

# TODO: compare with Nilearn functions and possibly extend


def load_mesh_geometry(surf_mesh):
    '''
    Load a mesh geometry into a dictionary with entries
    "coords" and "faces"

    Parameters
    ----------
    surf_mesh:
        Mesh geometry to be loaded, can be a path to a file
        (currently supported formats are freesurfer geometry formats,
        gii and ASCII-coded vtk, ply or obj) or a dictionary with the
        keys "coords" and "faces"

    Returns
    ----------
    dict
        Dictionary with a numpy array with key "coords" for a Numpy array of
        the x-y-z coordinates of the mesh vertices and key "faces" for a
        Numpy array of the the indices (into coords) of the mesh faces

    Notes
    ----------
    Originally created as part of Laminar Python [1]_

    References
    -----------
    .. [1] Huntenburg et al. (2017), Laminar Python: Tools for cortical
       depth-resolved analysis of high-resolution brain imaging data in
       Python. DOI: 10.3897/rio.3.e12346
    '''
    # if input is a filename, try to load it with nibabel
    if isinstance(surf_mesh, basestring):
        if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                surf_mesh.endswith('inflated')):
            coords, faces = nb.freesurfer.io.read_geometry(surf_mesh)
        elif surf_mesh.endswith('gii'):
            coords, faces = nb.gifti.read(surf_mesh).getArraysFromIntent(
                nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data, \
                nb.gifti.read(surf_mesh).getArraysFromIntent(
                nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
        elif surf_mesh.endswith('vtk'):
            coords, faces, _ = _read_vtk(surf_mesh)
        elif surf_mesh.endswith('ply'):
            coords, faces = _read_ply(surf_mesh)
        elif surf_mesh.endswith('obj'):
            coords, faces = _read_obj(surf_mesh)
        else:
            raise ValueError('Currently supported file formats are freesurfer '
                             'geometry formats and gii, vtk, ply, obj')
    elif isinstance(surf_mesh, dict):
        if ('faces' in surf_mesh and 'coords' in surf_mesh):
            coords, faces = surf_mesh['coords'], surf_mesh['faces']
        else:
            raise ValueError('If surf_mesh is given as a dictionary it '
                             'must contain items with keys "coords" and '
                             '"faces"')
    else:
        raise ValueError('Input surf_mesh must be a either filename or a '
                         'dictionary containing items with keys "coords" '
                         'and "faces"')
    return {'coords': coords, 'faces': faces}


def load_mesh_data(surf_data, gii_darray=None):
    '''
    Loads mesh data into a Numpy array

    Parameters
    ----------
    surf_data:
        Mesh data to be loaded, can be a Numpy array or a path to a file.
        Currently supported formats are freesurfer data formats (mgz, curv,
        sulc, thickness, annot, label), nii, gii, ASCII-coded vtk and txt
    gii_darray: int, optional
        Index of gii data array to load (default is to load all)

    Returns
    ----------
    np.ndarray
        Numpy array containing the data

    Notes
    ----------
    Originally created as part of Laminar Python [1]_

    References
    -----------
    .. [1] Huntenburg et al. (2017), Laminar Python: Tools for cortical
       depth-resolved analysis of high-resolution brain imaging data in
       Python. DOI: 10.3897/rio.3.e12346
    '''
    # if the input is a filename, load it
    if isinstance(surf_data, basestring):
        if (surf_data.endswith('nii') or surf_data.endswith('nii.gz') or
                surf_data.endswith('mgz')):
            data = np.squeeze(nb.load(surf_data).get_data())
        elif (surf_data.endswith('curv') or surf_data.endswith('sulc') or
                surf_data.endswith('thickness')):
            data = nb.freesurfer.io.read_morph_data(surf_data)
        elif surf_data.endswith('annot'):
            data = nb.freesurfer.io.read_annot(surf_data)[0]
        elif surf_data.endswith('label'):
            data = nb.freesurfer.io.read_label(surf_data)
        # check if this works with multiple indices (if dim(data)>1)
        elif surf_data.endswith('gii'):
            fulldata = nb.gifti.giftiio.read(surf_data)
            n_vectors = len(fulldata.darrays)
            if n_vectors == 1:
                data = fulldata.darrays[0].data
            else:
                if gii_darray is not None:
                    data = fulldata.darrays[gii_darray].data
                else:
                    print('Multiple gii data arrays found and gii_darray is '
                          'not set, output will be a matrix')
                    data = np.zeros([len(fulldata.darrays[gii_darray].data),
                                     n_vectors])
                    for gii_darray in range(n_vectors):
                        data[:, gii_darray] = fulldata.darrays[gii_darray].data
        elif surf_data.endswith('vtk'):
            _, _, data = _read_vtk(surf_data)
        elif surf_data.endswith('txt'):
            data = np.loadtxt(surf_data)
        else:
            raise ValueError('Format of data file not recognized. Currently '
                             'supported formats are freesurfer data formats '
                             '(mgz, sulc, curv, thickness, annot, label)'
                             'nii', 'gii, ASCII-coded vtk and txt')
    elif isinstance(surf_data, np.ndarray):
        data = np.squeeze(surf_data)
    return data


def save_mesh_data(filename, surf_data):
    '''
    Saves surface data that is a Numpy array to file

    Parameters
    ----------
    filename: str
        Full path and filename under which surfaces data should be saved. The
        extension determines the file format. Currently supported are
        freesurfer formats curv, thickness, sulc and ASCII-coded txt'
    surf_data: np.ndarray
        Surface data to be saved

    Notes
    ----------
    Originally created as part of Laminar Python [1]_

    References
    -----------
    .. [1] Huntenburg et al. (2017), Laminar Python: Tools for cortical
       depth-resolved analysis of high-resolution brain imaging data in
       Python. DOI: 10.3897/rio.3.e12346
    '''
    if isinstance(filename, basestring) and isinstance(surf_data, np.ndarray):
        if (filename.endswith('curv') or filename.endswith('thickness') or
                filename.endswith('sulc')):
            nb.freesurfer.io.write_morph_data(filename, surf_data)
            print("\nSaving {0}").format(filename)
        elif filename.endswith('txt'):
            np.savetxt(filename, surf_data)
            print("\nSaving {0}").format(filename)
        else:
            raise ValueError('File format not recognized. Currently supported '
                             'are freesurfer formats curv, sulc, thickness '
                             'and ASCII coded txt')
    else:
        raise ValueError('Filename must be a string')


def save_mesh_geometry(filename, surf_dict):
    '''
    Saves surface mesh geometry to file

    Parameters
    ----------
    filename: str
        Full path and filename under which surfaces data should be saved. The
        extension determines the file format. Currently supported are
        freesurfer geometry formats, gii and ASCII-coded vtk, obj, ply'
    surf_dict: dict
        Surface mesh geometry to be saved. Dictionary with a numpy array with
        key "coords" for a Numpy array of the x-y-z coordinates of the mesh
        vertices and key "faces2 for a Numpy array of the the indices
        (into coords) of the mesh faces

    Notes
    ----------
    Originally created as part of Laminar Python [1]_

    References
    -----------
    .. [1] Huntenburg et al. (2017), Laminar Python: Tools for cortical
       depth-resolved analysis of high-resolution brain imaging data in
       Python. DOI: 10.3897/rio.3.e12346
    '''
    if isinstance(filename, basestring) and isinstance(surf_dict, dict):
        if (filename.endswith('orig') or filename.endswith('pial') or
                filename.endswith('white') or filename.endswith('sphere') or
                filename.endswith('inflated')):
            nb.freesurfer.io.write_geometry(filename, surf_dict['coords'],
                                            surf_dict['faces'])
            print("\nSaving {0}").format(filename)
        elif filename.endswith('gii'):
            _write_gifti(filename, surf_dict['coords'], surf_dict['faces'])
            print("\nSaving {0}").format(filename)
        elif filename.endswith('vtk'):
            if 'data' in surf_dict.keys():
                _write_vtk(filename, surf_dict['coords'], surf_dict['faces'],
                           surf_dict['data'])
                print("\nSaving {0}").format(filename)
            else:
                _write_vtk(filename, surf_dict['coords'], surf_dict['faces'])
                print("\nSaving {0}").format(filename)
        elif filename.endswith('ply'):
            _write_ply(filename, surf_dict['coords'], surf_dict['faces'])
            print("\nSaving {0}").format(filename)
        elif filename.endswith('obj'):
            _write_obj(filename, surf_dict['coords'], surf_dict['faces'])
            print("\nSaving {0}").format(filename)
            print('To view mesh in brainview, run the command:\n')
            print('average_objects ' + filename + ' ' + filename)
    else:
        raise ValueError('Filename must be a string and surf_dict must be a '
                         'dictionary with keys "coords" and "faces"')


# function to read vtk files
# ideally use pyvtk, but it didn't work for our data, look into why
def _read_vtk(file):
    '''
    Reads ASCII coded vtk files using pandas,
    returning vertices, faces and data as three numpy arrays.
    '''
    import pandas as pd
    import csv
    # read full file while dropping empty lines
    try:
        vtk_df = pd.read_csv(file, header=None, engine='python')
    except csv.Error:
        raise ValueError(
            'This vtk file appears to be binary coded currently only ASCII '
            'coded vtk files can be read')
    vtk_df = vtk_df.dropna()
    # extract number of vertices and faces
    number_vertices = int(vtk_df[vtk_df[0].str.contains(
                                            'POINTS')][0].iloc[0].split()[1])
    number_faces = int(vtk_df[vtk_df[0].str.contains(
                                            'POLYGONS')][0].iloc[0].split()[1])
    # read vertices into df and array
    start_vertices = (vtk_df[vtk_df[0].str.contains(
                                            'POINTS')].index.tolist()[0]) + 1
    vertex_df = pd.read_csv(file, skiprows=range(start_vertices),
                            nrows=number_vertices, sep='\s*',
                            header=None, engine='python')
    if np.array(vertex_df).shape[1] == 3:
        vertex_array = np.array(vertex_df)
    # sometimes the vtk format is weird with 9 indices per line,
    # then it has to be reshaped
    elif np.array(vertex_df).shape[1] == 9:
        vertex_df = pd.read_csv(file, skiprows=range(start_vertices),
                                nrows=int(number_vertices / 3) + 1,
                                sep='\s*', header=None, engine='python')
        vertex_array = np.array(vertex_df.iloc[0:1, 0:3])
        vertex_array = np.append(vertex_array, vertex_df.iloc[0:1, 3:6],
                                 axis=0)
        vertex_array = np.append(vertex_array, vertex_df.iloc[0:1, 6:9],
                                 axis=0)
        for row in range(1, (int(number_vertices / 3) + 1)):
            for col in [0, 3, 6]:
                vertex_array = np.append(vertex_array, np.array(
                    vertex_df.iloc[row:(row + 1), col:(col + 3)]), axis=0)
        # strip rows containing nans
        vertex_array = vertex_array[~np.isnan(vertex_array)].reshape(
                                                            number_vertices, 3)
    else:
        print "vertex indices out of shape"
    # read faces into df and array
    start_faces = (vtk_df[vtk_df[0].str.contains(
                                            'POLYGONS')].index.tolist()[0]) + 1
    face_df = pd.read_csv(file, skiprows=range(start_faces),
                          nrows=number_faces, sep='\s*', header=None,
                          engine='python')
    face_array = np.array(face_df.iloc[:, 1:4])
    # read data into df and array if exists
    if vtk_df[vtk_df[0].str.contains('POINT_DATA')].index.tolist() != []:
        start_data = (vtk_df[vtk_df[0].str.contains(
                                        'POINT_DATA')].index.tolist()[0]) + 3
        number_data = number_vertices
        data_df = pd.read_csv(file, skiprows=range(start_data),
                              nrows=number_data, sep='\s*', header=None,
                              engine='python')
        data_array = np.array(data_df)
    else:
        data_array = np.empty(0)

    return vertex_array, face_array, data_array


def _read_ply(file):
    import pandas as pd
    import csv
    # read full file and drop empty lines
    try:
        ply_df = pd.read_csv(file, header=None, engine='python')
    except csv.Error:
        raise ValueError(
            'This ply file appears to be binary coded currently only '
            'ASCII coded ply files can be read')
    ply_df = ply_df.dropna()
    # extract number of vertices and faces, and row that marks end of header
    number_vertices = int(ply_df[ply_df[0].str.contains(
                                    'element vertex')][0].iloc[0].split()[2])
    number_faces = int(ply_df[ply_df[0].str.contains(
                                    'element face')][0].iloc[0].split()[2])
    end_header = ply_df[ply_df[0].str.contains('end_header')].index.tolist()[0]
    # read vertex coordinates into dict
    vertex_df = pd.read_csv(file, skiprows=range(end_header + 1),
                            nrows=number_vertices, sep='\s*', header=None,
                            engine='python')
    vertex_array = np.array(vertex_df)
    # read face indices into dict
    face_df = pd.read_csv(file,
                          skiprows=range(end_header + number_vertices + 1),
                          nrows=number_faces, sep='\s*', header=None,
                          engine='python')
    face_array = np.array(face_df.iloc[:, 1:4])

    return vertex_array, face_array


# function to read MNI obj mesh format
def _read_obj(file):

    def chunks(l, n):
        """Yield n-sized chunks from l"""
        for i in xrange(0, len(l), n):
            yield l[i:i + n]

    def indices(lst, element):
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset + 1)
            except ValueError:
                return result
            result.append(offset)
    fp = open(file, 'r')
    n_vert = []
    n_poly = []
    k = 0
    Polys = []
    # Find number of vertices and number of polygons, stored in .obj file.
    # Then extract list of all vertices in polygons
    for i, line in enumerate(fp):
        if i == 0:
            # Number of vertices
            n_vert = int(line.split()[6])
            XYZ = np.zeros([n_vert, 3])
        elif i <= n_vert:
            XYZ[i - 1] = map(float, line.split())
        elif i > 2 * n_vert + 5:
            if not line.strip():
                k = 1
            elif k == 1:
                Polys.extend(line.split())
    Polys = map(int, Polys)
    npPolys = np.array(Polys)
    triangles = np.array(list(chunks(Polys, 3)))
    return XYZ, triangles


def _write_gifti(surf_mesh, coords, faces):
    coord_array = nb.gifti.GiftiDataArray(data=coords,
                                          intent=nb.nifti1.intent_codes[
                                              'NIFTI_INTENT_POINTSET'])
    face_array = nb.gifti.GiftiDataArray(data=faces,
                                         intent=nb.nifti1.intent_codes[
                                             'NIFTI_INTENT_TRIANGLE'])
    gii = nb.gifti.GiftiImage(darrays=[coord_array, face_array])
    nb.gifti.write(gii, surf_mesh)


def _write_obj(surf_mesh, coords, faces):
    # write out MNI - obj format
    n_vert = len(coords)
    XYZ = coords.tolist()
    Tri = faces.tolist()
    with open(surf_mesh, 'w') as s:
        line1 = "P 0.3 0.3 0.4 10 1 " + str(n_vert) + "\n"
        s.write(line1)
        k = -1
        for a in XYZ:
            k += 1
            cor = ' ' + ' '.join(map(str, XYZ[k]))
            s.write('%s\n' % cor)
        s.write('\n')
        for a in XYZ:
            s.write(' 0 0 0\n')
        s.write('\n')
        l = ' ' + str(len(Tri)) + '\n'
        s.write(l)
        s.write(' 0 1 1 1 1\n')
        s.write('\n')
        nt = len(Tri) * 3
        Triangles = np.arange(3, nt + 1, 3)
        Rounded8 = np.shape(Triangles)[0] / 8
        N8 = 8 * Rounded8
        Triangles8 = Triangles[0:N8]
        RowsOf8 = np.split(Triangles8, N8 / 8)
        for r in RowsOf8:
            L = r.tolist()
            Lint = map(int, L)
            Line = ' ' + ' '.join(map(str, Lint))
            s.write('%s\n' % Line)
        L = Triangles[N8:].tolist()
        Lint = map(int, L)
        Line = ' ' + ' '.join(map(str, Lint))
        s.write('%s\n' % Line)
        s.write('\n')
        ListOfTriangles = np.array(Tri).flatten()
        Rounded8 = np.shape(ListOfTriangles)[0] / 8
        N8 = 8 * Rounded8
        Triangles8 = ListOfTriangles[0:N8]
        ListTri8 = ListOfTriangles[0:N8]
        RowsOf8 = np.split(Triangles8, N8 / 8)
        for r in RowsOf8:
            L = r.tolist()
            Lint = map(int, L)
            Line = ' ' + ' '.join(map(str, Lint))
            s.write('%s\n' % Line)
        L = ListOfTriangles[N8:].tolist()
        Lint = map(int, L)
        Line = ' ' + ' '.join(map(str, Lint))
        s.write('%s\n' % Line)


def _write_vtk(filename, vertices, faces, data=None, comment=None):
    '''
    Creates ASCII coded vtk file from numpy arrays using pandas.
    Inputs:
    -------
    (mandatory)
    * filename: str, path to location where vtk file should be stored
    * vertices: numpy array with vertex coordinates,  shape (n_vertices, 3)
    * faces: numpy array with face specifications, shape (n_faces, 3)
    (optional)
    * data: numpy array with data points, shape (n_vertices, n_datapoints)
        NOTE: n_datapoints can be =1 but cannot be skipped (n_vertices,)
    * comment: str, is written into the comment section of the vtk file
    Usage:
    ---------------------
    _write_vtk('/path/to/vtk/file.vtk', v_array, f_array)
    '''

    import pandas as pd
    # infer number of vertices and faces
    number_vertices = vertices.shape[0]
    number_faces = faces.shape[0]
    if data is not None:
        number_data = data.shape[0]
    # make header and subheader dataframe
    header = ['# vtk DataFile Version 3.0',
              '%s' % comment,
              'ASCII',
              'DATASET POLYDATA',
              'POINTS %i float' % number_vertices
              ]
    header_df = pd.DataFrame(header)
    sub_header = ['POLYGONS %i %i' % (number_faces, 4 * number_faces)]
    sub_header_df = pd.DataFrame(sub_header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(vertices)
    # make dataframe from faces, appending first row of 3's
    # (indicating the polygons are triangles)
    triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
    triangles = triangles.astype(int)
    faces = faces.astype(int)
    faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False, float_format='%.3f',
                         sep=' ')
    with open(filename, 'a') as f:
        sub_header_df.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False, float_format='%.0f',
                        sep=' ')
    # if there is data append second subheader and data
    if data is not None:
        datapoints = data.shape[1]
        sub_header2 = ['POINT_DATA %i' % (number_data),
                       'SCALARS EmbedVertex float %i' % (datapoints),
                       'LOOKUP_TABLE default']
        sub_header_df2 = pd.DataFrame(sub_header2)
        data_df = pd.DataFrame(data)
        with open(filename, 'a') as f:
            sub_header_df2.to_csv(f, header=False, index=False)
        with open(filename, 'a') as f:
            data_df.to_csv(f, header=False, index=False, float_format='%.16f',
                           sep=' ')


def _write_ply(filename, vertices, faces, comment=None):
    import pandas as pd
    print "writing ply format"
    # infer number of vertices and faces
    number_vertices = vertices.shape[0]
    number_faces = faces.shape[0]
    # make header dataframe
    header = ['ply',
              'format ascii 1.0',
              'comment %s' % comment,
              'element vertex %i' % number_vertices,
              'property float x',
              'property float y',
              'property float z',
              'element face %i' % number_faces,
              'property list uchar int vertex_indices',
              'end_header'
              ]
    header_df = pd.DataFrame(header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(vertices)
    # make dataframe from faces, adding first row of 3s (indicating triangles)
    triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
    triangles = triangles.astype(int)
    faces = faces.astype(int)
    faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False,
                         float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False,
                        float_format='%.0f', sep=' ')
