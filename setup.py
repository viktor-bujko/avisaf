#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Charles University

from setuptools import setup, find_packages

requirements = list(map(str.strip, open("./requirements.txt", mode="r").readlines()))

setup(
    author="Viktor Bujko",
    author_email="viktorbjk@gmail.com",
    name="avisaf",
    version="1.0",
    description="Aviation safety report Named Entity Recognizer",
    long_description="",
    classifiers=[
        "Programming Language :: Python :: 3.8.2",
    ],
    keywords=["named entity recognition", "NER", "aviation", "safety"],
    url="https://github.com/viktor-bujko/avisaf",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    packages=find_packages(where="avisaf"),
    # packages=["avisaf", "avisaf.training", "avisaf.classification", "avisaf.util"],
    entry_points={
        "console_scripts": [
            "avisaf=avisaf.main:main",
        ],
    },
)
