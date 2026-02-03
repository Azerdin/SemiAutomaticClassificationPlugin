@echo off

if "%~1"=="" (
    echo Missing disk letter
    exit /b 1
)

if "%~2"=="" (
    echo Missing user name
    exit /b 1
)

set "DRIVE=%~1"
set "USER=%~2"

set "DEST=%DRIVE%:\Users\%USER%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\SemiAutomaticClassificationPlugin"

if exist "%DEST%" (
    rmdir /s /q "%DEST%"
)

mkdir "%DEST%"

robocopy "%cd%" "%DEST%" /E /R:2 /W:1

exit /b 0
