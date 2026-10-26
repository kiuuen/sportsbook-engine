@echo off
setlocal
set "BASE=C:\Users\kiuuen\Documents\sportsbook_engine"
cd /d "%BASE%"

if not exist "%BASE%\.env" (
  echo No se encontro %BASE%\.env
  pause
  exit /b 1
)

if not exist "%BASE%\config.json" (
  echo No se encontro %BASE%\config.json
  pause
  exit /b 1
)

rem Carga cada linea KEY=VALUE del .env (ignora lineas que empiezan con #)
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%BASE%\.env") do (
  set "%%A=%%~B"
)
npx -y @maximhq/bifrost -app-dir "%BASE%"
pause