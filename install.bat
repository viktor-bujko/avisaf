@ECHO OFF
if not exist venv (
    ECHO Creating virtual environment
    python -m venv venv
)
.\venv\Scripts\pip.exe install -U pip
ECHO Installing dependencies
.\venv\Scripts\pip.exe install -r requirements.txt
ECHO Installing spacy language model
.\venv\Scripts\python.exe -m spacy download en_core_web_md
ECHO Install avisaf tool
.\venv\Scripts\python.exe .\setup.py build && .\venv\Scripts\python.exe .\setup.py install
ECHO Now you can use ".\venv\Scripts\avisaf" command or ".\venv\Scripts\python.exe -m avisaf.main".
.\venv\Scripts\avisaf
PAUSE