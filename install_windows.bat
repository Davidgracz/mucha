@echo off
py -3.12 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy /Y .env.example .env >nul
python tools\make_demo_connectome.py --output data\connectome
 echo.
 echo GOTOWE. Uzupelnij DISCORD_TOKEN w pliku .env.
 echo Demo connectome jest tylko do testu. Instrukcja FlyWire jest w README.md.
pause
