@echo off


IF /I "%1"=="install" GOTO install
IF /I "%1"=="venv" GOTO venv
IF /I "%1"=="clean" GOTO clean
GOTO error

:install
	CALL make.bat venv
	@.\venv\Scripts\pip.exe install -U pip
	@.\venv\Scripts\pip.exe install -r requirements.txt
	@.\venv\Scripts\python.exe -m spacy download en_core_web_md
	@.\venv\Scripts\python.exe ./setup.py build && .\venv\Scripts\python.exe ./setup.py install
	GOTO :EOF

:venv
	if not exist venv (
		ECHO "Creating virtual environment"
		python -m venv venv
	)
	GOTO :EOF

:clean
	@echo "Cleaning directory"
	@rmdir /s /q avisaf.egg-info build dist venv && echo "Cleaning complete"
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF
