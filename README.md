## Distributing

Recently I tried to learn about [distributing Python packages through PyPI](https://python-packaging-user-guide.readthedocs.io/tutorials/distributing-packages/). While that seems more or less straightforward for pure Python projects, the story gets more complicated for our case, where extensions in another language need to be compiled in a platform-dependent manner. I ended up with some insights, but also a lot of remaining confusion...

Generally there are two options: 
1. compiling the extensions during installation
2. distributing pre-compiled binaries

For 1. I found very little information on how to actually implement that. One option seems to [build the extensions using distutils](https://docs.python.org/2/extending/building.html) in the setup.py script. Although I am not sure if that requires the user to install from source, instead of installing using e.g. pip. One [post](https://stackoverflow.com/questions/31380578/how-to-avoid-building-c-library-with-my-python-package) mentions that they used `build_ext` with pip to build a C extension during installation, but I could just not find any information on how exactly that works.

I found much more information on option 2. The preferred way to package binary extensions through are [wheels](https://python-packaging-user-guide.readthedocs.io/tutorials/distributing-packages/#wheels). 
>A wheel is a built package that can be installed without needing to go through the build process. Installing wheels is substantially faster for the end user than installing from a source distribution.

In our case we would need to create a platform wheel, which is a platform dependent wheel that contains compiled extensions. PyPI currently supports wheels for Windows and OS X but only a compatible [subset of linux distributions](https://www.python.org/dev/peps/pep-0513/). Wheels appear to be the preferred and more standardized way of packaging versus the older egg format (see this [discussion](https://packaging.python.org/discussions/wheel-vs-egg/)). This is also how e.g. [Numpy](https://pypi.python.org/pypi/numpy) distribution is managed.

#### Conda
Another option is to use conda to package our project. This is where I am still confused. Building a [conda package from an existing PyPI project](https://conda.io/docs/build_tutorials/pkgs.html) seems straightforward. Those can apparently be build for one platfrom, converted for others and then uploaded to anaconda.org. But I am not sure if that is only true for Python packages, or for binary extensions too, and how it works in the latter case (do the extensions already have to be on PyPI?). There is also a tutorial how to build a [conda package from scratch for any language](https://conda.io/docs/build_tutorials/postgis.html), for which no PyPI project is required. But it is still a bit obscure to me how exactly that works, and what are the advantages over using PyPI with wheels. I found some thoughts on the latter question [here under Myth#6](https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/)

Hopefully I can update with a better understanding soon. 


## To Docker or not to Docker

Reaching out to the neuroimaging community regarding means of software distribution got me many helpful replies and I learned quite a bit about Docker and related tools along the way. My main take-aways:

* Docker as a scientific tool in the neuroimaging community is of growing importance, as exemplified by [BIDS Apps](http://bids-apps.neuroimaging.io/) or [CBRAIN](http://natacha-beck.github.io/cbrain_docker/#/). However, it still in an early stage so that widespread adoption will take time and installation procedures might need to be tweaked due to changes to the Docker project itself.
* Default Docker images such as AlpineLinux do not add a lot of overhead (they are much smaller than virtual machines) and should not result in a significant performance penalties unless one is dealing with super efficient numerical code.
* Using Docker in combination with [Singularity](http://singularity.lbl.gov/) helps to run containers without root access and without Docker being actually installed, e.g. on HPCs
* When distributing through Docker it is even more important to have good regression tests across platforms and software versions. There are two projects trying to make this easier: [neurodocker](https://github.com/kaczmarj/neurodocker) and [niceman](https://github.com/ReproNim/niceman)
* One interesting approach is, in addition to the Docker image, to publish a [standalone script](https://github.com/poldracklab/fmriprep/blob/master/wrapper/fmriprep_docker.py) that generates Docker commands for users through [PyPI](https://pypi.python.org/pypi/fmriprep-docker) , so it can be install with pip.

One of the most helpful comments regarding our project I found to be this ([on neurostars.org](https://neurostars.org/t/using-docker-to-distribute-highres-neuroimaging-software/442/2?u=juhuntenburg)):

>"If you are looking to distribute a library that is intended to be integrated with other software wrapping it in Docker will make it hard if not impossible. If you are looking to distribute a command line tool with complex set of dependencies Docker is a good fit." 

From the replies and talking to my mentors I got the sense that it will be the best choice for us to focus on more traditional ways of download and installation for now. Especially, because integrating our tools with other software is a main objective of this project. But we will also explore if it makes sense to additionally provide a Docker Image later on.

*Thanks for all the helpful comments and suggestions!*


## Wisdom of the crowd

This week, GSoC started with the "community bonding" phase. I already feel pretty bonded with my software community (in fact I constitute 1/3rd of it), so I use the time to reach out to the larger neuroimaging community regarding one of the first issues I want to address: 

*What is the best way to distribute the Python version of CBS Tools?*

I have previously discussed this question with my mentors and other colleagues, and one suggestion that came up is to deploy the tools via [Docker](https://www.docker.com/). In order to find out what neuroimaging folks think about docker I have started discussion threads on [neurostars](https://neurostars.org/t/using-docker-to-distribute-highres-neuroimaging-software/442) and in the [brainhack slack team](https://brainhack-slack-invite.herokuapp.com/). 

I know very little about docker myself, so I also started watching some [tutorials](https://www.youtube.com/playlist?list=PLoYCgNOIyGAAzevEST2qm2Xbe3aeLFvLc). Whether we end up using it or not, my feeling is it won't hurt in the future to know a thing or two about docker. 


## Google Summer of Code (GSoC)

On this site I will document my work during the Google Summer of Code 2017 with INCF. The fulll project proposal can be found [here](https://docs.google.com/document/d/1lkcTpcYT1r1qwh4GwccyWjY3cq2VZ89AlQoKa4Fd2aQ/edit?usp=sharing).


