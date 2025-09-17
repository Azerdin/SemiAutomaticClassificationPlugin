@ECHO OFF

if "%1"=="" (
    echo "Missing parameter with user name"
    exit /b 1
)

set "pluginPath=C:\Users\%1\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins"

if exist "%pluginPath%" (
    rmdir /s /q "%pluginPath%"
)

xcopy /E /I /Y "%CD%\.." "%pluginPath%"

exit /b 0