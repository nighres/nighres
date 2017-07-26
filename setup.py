from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
long_description = 'load from README'

setup(
    name='nires',
    version='0.1.0',
    description='Processing tools for high-resolution neuroimaging',
    long_description=long_description,
    url='TODO',
    author='Julia M Huntenburg, Pierre-Louis Bazin, Chris Steele',
    author_email='ju.huntenburg@gmail.com',
    license='TODO',
    classifiers=[
                 'Development Status :: 3 - Alpha',
                 'Intended Audience :: Researchers',
                 'Topic :: High-resolution Neuroimage Processing ',
                 'License :: TODO',
                 'Programming Language :: Python :: 2.7',
                 ],
    keywords='MRI high-resolution laminar',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'nibabel'],
)
