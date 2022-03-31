# CONTRIBUTING

## Docker

The base recipe was generated using
[neurodocker](https://github.com/ReproNim/neurodocker) from the Makefile.

If possible do not modify the Dockerfile directly but modify it via the
Makefile.

Creating the Dockerfile, building it and testing it can be done with:

```bash
make docker_test
```

The test only run examples 1, 2 and 3 from the `examples` folder.

## Dependencies version

The `conda-nighres.yml` specifies the version of ALL the dependencies.

This can be generated with `conda env export > conda-nighres.yml`.

Note that if you do this you will need to 

The minimalist version of this file would look like this:

```yml
name: nighres
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - jcc
  - Nilearn
  - gcc_linux-64
  - gxx_linux-64
```

If you need to update the following packages, do it in the
[`setup.py`](./setup.py) file:

- `numpy`
- `nibabel`
- `psutil`
- `antspyx`

Then run to update `conda-nighres.yml`

```bash
pip install .
conda env export > conda-nighres.yml
```

The following pacakages that are necessary for setting up the environment
building nighres JAVA or for running examples can updated directly.

- `pip`
- `jcc`
- `Nilearn`
- `gcc_linux-64`
- `gxx_linux-64`

This can be done with:

```bash
conda update -n nighres pip jcc Nilearn gcc_linux-64 gxx_linux-64
pip install .
conda env export > conda-nighres.yml
```
