#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Charles University

from setuptools import setup

requirements = list(map(str.strip, open('./requirements.txt', mode='r').readlines()))

setup(
    author='Viktor Bujko',
    email='viktorbjk@gmail.com',
    name='avisaf',
    version='1.0',
    description='Aviation safety report Named Entity Recognizer',
    long_description='',
    classifiers=[
        'Programming Language :: Python :: 3.6.9',
    ],
    keywords=['named entity recognition', 'NER', 'aviation', 'safety'],
    url='',
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    packages=[
        'avisaf.main',
        'avisaf.training',
        'avisaf.util'
    ],
    entry_points={
        'console_scripts': [
            'avisaf=avisaf.main.main:main',
        ],
    },
)
