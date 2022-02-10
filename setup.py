#!/usr/bin/env python

from setuptools import setup

__slrkit_version__ = ''

exec(open('slrkit/version.py').read())

setup(
    name='slrkit',
    version=__slrkit_version__
)
