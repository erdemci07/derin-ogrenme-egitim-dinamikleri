$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Tez Calismasi - Windows Otomatik Kurulum" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

function Test-VisualCppRuntime {
    $paths = @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    )

    foreach ($path in $paths) {
        if (Test-Path $path) {
            try {
                $installed = (Get-ItemProperty $path -ErrorAction Stop).Installed
                if ($installed -eq 1) {
                    return $true
                }
            }
            catch {
            }
        }
    }

    return $false
}

function Install-VisualCppRuntime {
    Write-Host "[0/7] Microsoft Visual C++ Redistributable kontrol ediliyor..." -ForegroundColor Yellow

    if (Test-VisualCppRuntime) {
        Write-Host "      Visual C++ Redistributable zaten kurulu." -ForegroundColor Green
        return
    }

    Write-Host "      Visual C++ Redistributable bulunamadi." -ForegroundColor Yellow
    Write-Host "      Resmi Microsoft kurulum dosyasi indiriliyor..." -ForegroundColor Yellow

    $installer = Join-Path $env:TEMP "vc_redist.x64.exe"
    $url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing

    Write-Host "      Kurulum baslatiliyor. Windows UAC onayi isteyebilir." -ForegroundColor Yellow
    $process = Start-Process -FilePath $installer -ArgumentList "/install", "/quiet", "/norestart" -Verb RunAs -Wait -PassThru

    if ($process.ExitCode -ne 0 -and $process.ExitCode -ne 3010) {
        throw "Visual C++ Redistributable kurulumu basarisiz oldu. Cikis kodu: $($process.ExitCode)"
    }

    Remove-Item $installer -Force -ErrorAction SilentlyContinue

    if (-not (Test-VisualCppRuntime)) {
        Write-Host "      Kurulum tamamlandi ancak kayit anahtari hemen gorunmedi." -ForegroundColor Yellow
        Write-Host "      Bilgisayari yeniden baslatmak gerekebilir." -ForegroundColor Yellow
    }
    else {
        Write-Host "      Visual C++ Redistributable kuruldu." -ForegroundColor Green
    }
}

function Bul-Python {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @("py", "-3")
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }

    throw "Python bulunamadi. Once Python 3.10+ kurup tekrar deneyin."
}

Install-VisualCppRuntime

$pythonKomutu = Bul-Python
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[1/7] Sanal ortam olusturuluyor (.venv)..." -ForegroundColor Yellow
    if ($pythonKomutu.Count -eq 2) {
        & $pythonKomutu[0] $pythonKomutu[1] -m venv (Join-Path $PSScriptRoot ".venv")
    }
    else {
        & $pythonKomutu[0] -m venv (Join-Path $PSScriptRoot ".venv")
    }
}
else {
    Write-Host "[1/7] .venv zaten mevcut, yeniden kullaniliyor." -ForegroundColor Green
}

Write-Host "[2/7] pip guncelleniyor..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[3/7] Gerekli kutuphaneler kuruluyor..." -ForegroundColor Yellow
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "[4/7] GPU / CUDA destegi kontrol ediliyor..." -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    Write-Host "      NVIDIA GPU algilandi." -ForegroundColor Green
    & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

    Write-Host "      CPU PyTorch paketi varsa kaldiriliyor..." -ForegroundColor Yellow
    & $venvPython -m pip uninstall -y torch torchvision torchaudio

    Write-Host "      CUDA destekli PyTorch kuruluyor (cu126)..." -ForegroundColor Yellow
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
}
else {
    Write-Host "      NVIDIA GPU / nvidia-smi bulunamadi. CUDA kurulumu atlandi." -ForegroundColor Yellow
    Write-Host "      CUDA yalnizca desteklenen NVIDIA GPU ile kullanilabilir." -ForegroundColor Yellow
}

Write-Host "[5/7] PyTorch kurulumu kontrol ediliyor..." -ForegroundColor Yellow
$torchKontrol = "import sys; import torch; print('PyTorch OK:', torch.__version__)"
& $venvPython -c $torchKontrol

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "PyTorch DLL yukleme hatasi algilandi." -ForegroundColor Red
    Write-Host "Visual C++ Redistributable yeni kurulduysa bilgisayari yeniden baslatip" -ForegroundColor Yellow
    Write-Host "kurulum_windows.cmd dosyasini tekrar calistirin." -ForegroundColor Yellow
    exit 1
}

Write-Host "[6/7] Kutuphaneler kontrol ediliyor..." -ForegroundColor Yellow
& $venvPython (Join-Path $PSScriptRoot "kurulumu_kontrol_et.py")

Write-Host "[7/7] VS Code icin Python yolu hazirlandi." -ForegroundColor Yellow
Write-Host ""
Write-Host "Kurulum tamamlandi." -ForegroundColor Green
Write-Host ""
Write-Host "VS Code Python yorumlayicisi:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\python.exe"
Write-Host ""
Write-Host "Duman testini calistirmak icin:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe calistir_duman_testi.py"
Write-Host ""
