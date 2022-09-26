import os
import builtins
from setuptools import setup, find_packages

abspath = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []

builtins.__SETUP__ = True
import dfhalo
version = dfhalo.__version__

setup(

    name='dfhalo',

    version=version,

    description='Halo Assessment for single exposures taken by Dragonfly',

    long_description=long_description,

    url='https://github.com/NGC4676/DFHalo',

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy',

    packages=find_packages(include=['DFhalo','DFHalo.']),

    python_requires='>=3.7',

    install_requires=install_requires,

)
