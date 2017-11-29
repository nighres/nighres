from os import path
from setuptools import setup, find_packages
# from setuptools.command.develop import develop
# from setuptools.command.install import install
# from subprocess import check_call

here = path.abspath(path.dirname(__file__))
build_script = path.join(here, "build.sh")
with open('README.rst') as f:
    long_description = f.read()


# # These commands run the build.sh script during pip installation
# class PostDevelopCommand(develop):
#     """Post-installation for development mode."""
#     def run(self):
#         develop.run(self)
#         check_call(build_script)
#
#
# class PostInstallCommand(install):
#     """Post-installation for installation mode."""
#     def run(self):
#         install.run(self)
#         check_call(build_script)

setup(
    name='nighres',
    version='1.0.0b9',
    description='Processing tools for high-resolution neuroimaging',
    long_description=long_description,
    url='https://nighres.readthedocs.io/',
    author='Julia M Huntenburg, Pierre-Louis Bazin, Chris Steele',
    author_email='ju.huntenburg@gmail.com',
    license='Apache License, 2.0',
    classifiers=[
                 'Development Status :: 4 - Beta',
                 'Topic :: Scientific/Engineering',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Programming Language :: Python :: 2.7',
                 ],
    # cmdclass={
    #           'develop': PostDevelopCommand,
    #           'install': PostInstallCommand,
    #          },
    keywords='MRI high-resolution laminar',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'nibabel'],
)
