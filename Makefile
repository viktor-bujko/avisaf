.PHONY: install venv clean docs

install: setup.py venv avisaf/
	venv/bin/pip3 install -U pip
	venv/bin/pip3 install -r requirements.txt
	venv/bin/python3 -m spacy download en_core_web_md
	venv/bin/python3 ./setup.py build && venv/bin/python ./setup.py install
	@. venv/bin/activate

venv: requirements.txt
	test -d venv || python3 -m venv venv
	@. venv/bin/activate
	@which venv/bin/pip3
	@which venv/bin/python3

clean:
	@echo "Cleaning directory"
	@rm -rf avisaf.egg-info build dist venv && echo "Cleaning complete"

docs:
	sphinx-apidoc -o docs/ avisaf
	$(MAKE) -C docs html
