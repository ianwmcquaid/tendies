import setuptools

from distutils.core import setup

setup(name='tendies',
    version='1.0.0',
    description='',
    author='',
    author_email='',
    packages=['full_functionality'],
    url="https://github.com/pypa/sampleproject",
    classifiers=[
    "Programming Language :: Python :: 3.6",
    "License :: OSI Approved :: TODO",
    "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow==1.12.2',
        'keras==2.2.4'
    ]
)