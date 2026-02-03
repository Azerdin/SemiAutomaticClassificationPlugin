@ECHO OFF

if "%1"=="" (
    echo "Missing parameter with disk name"
    exit /b 1
)

if "%2"=="" (
    echo "Missing parameter with user name"
    exit /b 1
)

set "pluginPath=%1:\Users\%2\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins"

if exist "%pluginPath%" (
    rmdir /s /q "%pluginPath%"
)

xcopy /E /I /Y "%CD%\.." "%pluginPath%"

exit /b 0