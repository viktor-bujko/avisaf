.PHONY: install run venv clean docs

install: setup.py venv avisaf/
	venv/bin/python ./setup.py build && venv/bin/python ./setup.py install

venv: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate
	which venv/bin/pip
	which venv/bin/python
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt

run: setup.py avisaf/
	test -d venv && . venv/bin/activate

clean:
	@echo "Cleaning directory"
	@rm -rf avisaf.egg-info build dist venv && echo "Cleaning complete"

docs: install
	sphinx-apidoc -o docs/source src -f -e
	$(MAKE) -C docs html
