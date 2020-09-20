#!/bin/bash


if [ ! -d "./venv/" ]; then
	python3 -m venv venv
	echo "Created new virtual environment venv"
fi

. venv/bin/activate

pip install -U pip
pip install -r requirements.txt

./setup.py build && ./setup.py install

python -m spacy download en_core_web_md
