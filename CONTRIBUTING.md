# CONTRIBUTING

- [CONTRIBUTING](#contributing)
  - [Docker](#docker)
  - [Dependencies version](#dependencies-version)
  - [Java dependencies version](#java-dependencies-version)

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

The `conda-nighres.yml` specifies the version of the basic dependencies.


If you need to update the following packages, do it in the
[`setup.py`](./setup.py) file:

- `numpy`
- `nibabel`
- `psutil`
- `antspyx`


The following pacakages that are necessary for setting up the environment
building nighres JAVA or for running examples can be updated directly.

- `pip`
- `jcc`
- `Nilearn`


## Java dependencies version

The versions of the Java dependencies for nighres are in `dependencies_sha.sh`.

They contain the commit id (SHA1) from the cbstools and imcntk repo.

For example.

```bash
cbstools_sha=7a34255
imcntk_sha=ea901d8
```

Those values are then used by the build process (`build.sh`).

To update those to the latest commit on the master branch simply use
`make update_dep_shasum`.
