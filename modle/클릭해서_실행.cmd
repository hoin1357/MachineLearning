@echo off
setlocal

cd /d "%~dp0"
title ML3 Local Server
set "LOG=%~dp0run_server.log"

where py >nul 2>nul
if errorlevel 1 (
  goto try_python
) else (
  set "PYTHON_CMD=py"
  goto run
)

:try_python
where python >nul 2>nul
if errorlevel 1 (
  goto no_python
) else (
  set "PYTHON_CMD=python"
  goto run
)

:no_python
echo Python was not found.
echo Please install Python and run this file again.
pause
exit /b 1

:run
echo Starting ML3 web page.
echo Log file: %LOG%
echo.
echo [%date% %time%] Start > "%LOG%"

echo Checking Python packages...
%PYTHON_CMD% -c "import importlib.util, sys; mods=['fastapi','uvicorn','pandas','numpy','sklearn','catboost','korean_lunar_calendar','httpx']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('missing:', ', '.join(missing) if missing else 'none'); sys.exit(1 if missing else 0)" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Installing required packages. This can take a few minutes on first run.
  %PYTHON_CMD% -m pip install -r backend\requirements.txt >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo.
    echo Package installation failed.
    echo See run_server.log for details.
    echo.
    type "%LOG%"
    pause
    exit /b 1
  )
)

echo Starting server. Wait until the browser opens.
echo URL: http://127.0.0.1:8765
echo.
%PYTHON_CMD% scripts\run_demo.py >> "%LOG%" 2>&1

echo.
echo The server stopped or failed to start.
echo See run_server.log for details.
echo.
type "%LOG%"

echo.
echo Press any key to close this window.
pause >nul
