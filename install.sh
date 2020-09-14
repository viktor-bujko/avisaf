#!/bin/bash

./setup.py build && ./setup.py install

python -m spacy download en_core_web_md
