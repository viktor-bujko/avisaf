@echo off


IF /I "%1"=="install" GOTO install
IF /I "%1"=="venv" GOTO venv
IF /I "%1"=="clean" GOTO clean
IF /I "%1"=="docs" GOTO docs
GOTO error

:install
	CALL make.bat setup.py
	CALL make.bat venv
	CALL make.bat avisaf/
	venv/bin/pip3 install -U pip
	venv/bin/pip3 install -r requirements.txt
	venv/bin/python3 -m spacy download en_core_web_md
	venv/bin/python3 ./setup.py build && venv/bin/python ./setup.py install
	@. venv/bin/activate
	GOTO :EOF

:venv
	CALL make.bat requirements.txt
	test -d venv || python3 -m venv venv
	@. venv/bin/activate
	@which venv/bin/pip3
	@which venv/bin/python3
	GOTO :EOF

:clean
	@echo "Cleaning directory"
	@rm -rf avisaf.egg-info build dist venv && echo "Cleaning complete"
	GOTO :EOF

:docs
	sphinx-apidoc -o docs/ avisaf
	CALL make.bat -C docs html
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF
