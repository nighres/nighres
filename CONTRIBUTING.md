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
