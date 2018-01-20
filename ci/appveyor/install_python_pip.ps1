# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel and Kyle Kastner
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

# Abort the script on any failure
$ErrorActionPreference = "Stop"

$BASE_URL = "https://www.python.org/ftp/python/"

$GET_PIP_URL = "http://releases.numenta.org/pip/1ebd3cb7a5a3073058d0c9552ab074bd/get-pip.py"
$GET_PIP_PATH = "C:\get-pip.py"

function main () {
    $python_path = $env:PYTHONHOME + "/python.exe"
    $pip_path = $env:PYTHONHOME + "/Scripts/pip.exe"

    Write-Host "pip install " wheel==0.25.0
    & $pip_path install wheel==0.25.0

    Write-Host "pip install " boto
    & $pip_path install boto

    Write-Host "pip install " twine
    & $pip_path install twine

    Write-Host "pip install " numpy==1.12.1
    & $pip_path install numpy==1.12.1
}

main
