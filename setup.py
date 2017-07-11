
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
long_description = 'load from README.rst'

setup(
    name='cbstools-python',
    version='1.1.0',
    description='CBS High Resolution Image Processing Tools in Python',
    long_description=long_description,
    url='https://github.com/juhuntenburg/cbstools-python',
    author='Julia Huntenburg',
    author_email='ju.huntenburg@gmail.com',
    license='CC BY-SA 4.0',
    classifiers=[
                 'Development Status :: 3 - Alpha',
                 'Intended Audience :: Researchers',
                 'Topic :: High Resolution Image Processing ',
                 'License :: Creative Commons Attribution-ShareAlike 4.0 International License :: CC BY-SA 4.0',
                 'Programming Language :: Python :: 2.7',
                 ],
    keywords='MRI high-resolution laminar',
    packages=find_packages(),
    install_requires=['numpy', 'nibabel'],
)
