.PHONY: fetch_files install run venv clean docs

install: fetch_files setup.py venv avisaf/
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt
	venv/bin/python -m spacy download en_core_web_md
	venv/bin/python ./setup.py build && venv/bin/python ./setup.py install
	. venv/bin/activate

fetch_files:
	@echo "Downloading large data files"
	@echo "'git lfs' command is needed; install if needed and run the command again"
	@git lfs pull && echo "Files fetched successfully"

venv: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate
	which venv/bin/pip
	which venv/bin/python

run: setup.py avisaf/
	test -d venv && . venv/bin/activate

clean:
	@echo "Cleaning directory"
	@rm -rf avisaf.egg-info build dist venv && echo "Cleaning complete"

docs: install
	sphinx-apidoc -o docs/source src -f -e
	$(MAKE) -C docs html
