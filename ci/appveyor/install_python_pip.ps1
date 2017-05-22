# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel and Kyle Kastner
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

# Abort the script on any failure
$ErrorActionPreference = "Stop"

$BASE_URL = "https://www.python.org/ftp/python/"

$GET_PIP_URL = "http://releases.numenta.org/pip/1ebd3cb7a5a3073058d0c9552ab074bd/get-pip.py"
$GET_PIP_PATH = "C:\get-pip.py"


function DownloadPython ($python_version, $platform_suffix) {
    $webclient = New-Object System.Net.WebClient
    $filename = "python-" + $python_version + $platform_suffix + ".msi"
    $url = $BASE_URL + $python_version + "/" + $filename

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 5 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 3
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
   }
   Write-Host "File saved at" $filepath
   return $filepath
}


function InstallPython ($python_version, $architecture, $python_home) {
    Write-Host "Installing Python" $python_version "for" $architecture "bit architecture to" $python_home
    if ( $(Try { Test-Path $python_home.trim() } Catch { $false }) ) {
        Write-Host $python_home "already exists, skipping."
        return $false
    }
    if ($architecture -eq "32") {
        $platform_suffix = ""
    } else {
        $platform_suffix = ".amd64"
    }
    $filepath = DownloadPython $python_version $platform_suffix
    Write-Host "Installing" $filepath "to" $python_home
    $args = "/qn /i $filepath TARGETDIR=$python_home"
    Write-Host "msiexec.exe" $args
    Start-Process -FilePath "msiexec.exe" -ArgumentList $args -Wait -Passthru
    Write-Host "Python $python_version ($architecture) installation complete"
    return $true
}


function InstallPip ($python_home) {
    $pip_path = $python_home + "/Scripts/pip.exe"
    $python_path = $python_home + "/python.exe"
    if ( $(Try { Test-Path $pip_path.trim() } Catch { $false }) ) {
        Write-Host "pip already installed at " $pip_path ". Upgrading..."

        # Upgrade it to avoid error exit code during usage
        & $python_path -m pip install --upgrade pip

        return $false
    }

    Write-Host "Installing pip..."
    $webclient = New-Object System.Net.WebClient
    $webclient.DownloadFile($GET_PIP_URL, $GET_PIP_PATH)
    Write-Host "Executing:" $python_path $GET_PIP_PATH
    Start-Process -FilePath "$python_path" -ArgumentList "$GET_PIP_PATH" -Wait -Passthru
    return $true
}

function main () {
    InstallPython $env:PYTHON_VERSION $env:PYTHON_ARCH $env:PYTHONHOME
    InstallPip $env:PYTHONHOME

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
