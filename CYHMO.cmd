@echo off
rem CYHMO.cmd                -> instala se preciso e abre o mod
rem CYHMO.cmd --skip-doctor  -> abre sem o diagnostico de primeira execucao
rem CYHMO.cmd doctor         -> diagnostico do ambiente
rem CYHMO_NOPAUSE=1 desliga a pausa final, para uso em script.
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
rem O mod sai com este codigo depois de trocar os arquivos por uma versao nova: quem
rem carrega o codigo novo e o processo seguinte, nao ele mesmo.
set "UPDATED=10"
set "REINSTALL=.cyhmo-reinstall"
set "DOCTOR_OK=.cyhmo-doctor-ok"

set "SKIP_DOCTOR="
if /i "%~1"=="--skip-doctor" set "SKIP_DOCTOR=1"

call :install_if_needed
if errorlevel 1 (
  echo.
  call :hold
  exit /b 1
)

if "%~1"=="" goto :launch
if defined SKIP_DOCTOR goto :launch

"%PYTHON%" -m cyhmo %*
set "CODE=%ERRORLEVEL%"
goto :finish

:launch
call :check_environment
set "BROWSER="

:run
"%PYTHON%" -m cyhmo run %BROWSER%
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="%UPDATED%" goto :finish
rem A aba do navegador continua aberta na mesma URL e reconecta sozinha.
set "BROWSER=--no-browser"
call :install_after_update
goto :run

:finish
if "%CODE%"=="0" exit /b 0

echo.
if /i not "%~1"=="doctor" (
  echo Algo falhou. Para o diagnostico completo, rode:
  echo   CYHMO.cmd doctor
)
call :hold
exit /b %CODE%

:install_if_needed
if exist "%PYTHON%" (
  "%PYTHON%" -c "import cyhmo" >NUL 2>&1
  if not errorlevel 1 exit /b 0
)
set "FRESH=1"
echo Preparando o ambiente. Isso leva alguns minutos.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Pinned -FromLauncher
exit /b %ERRORLEVEL%

:install_after_update
if not exist "%REINSTALL%" exit /b 0
del "%REINSTALL%" >NUL 2>&1
echo A versao nova mudou as dependencias. Atualizando o ambiente.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Pinned -SkipBackend -FromLauncher
exit /b 0

:check_environment
rem O diagnostico roda uma vez so, e so com o emulador ja aberto: com o PCSX2 fechado
rem ele apenas listaria o que ainda nao da para checar, e assusta em vez de ajudar.
if defined SKIP_DOCTOR exit /b 0
if exist "%DOCTOR_OK%" exit /b 0
tasklist /NH 2>NUL | find /I "pcsx2" >NUL
if errorlevel 1 (
  call :warn_no_emulator
  exit /b 0
)
call :run_doctor
exit /b 0

:warn_no_emulator
if not defined FRESH exit /b 0
echo.
echo Abra o PCSX2 com o Lifeline rodando: o CYHMO espera pelo emulador e conecta
echo sozinho, sem precisar reabrir nada. Para conferir o ambiente quando quiser:
echo   CYHMO.cmd doctor
exit /b 0

:run_doctor
echo.
echo Conferindo o ambiente (so desta vez)...
"%PYTHON%" -m cyhmo doctor
if errorlevel 1 (
  echo.
  echo O diagnostico apontou as pendencias acima. O CYHMO abre mesmo assim.
  exit /b 0
)
> "%DOCTOR_OK%" echo ok
exit /b 0

:hold
if not "%CYHMO_NOPAUSE%"=="1" pause
exit /b 0
