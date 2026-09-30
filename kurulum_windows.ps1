$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Tez Calismasi - Windows Otomatik Kurulum" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

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

$pythonKomutu = Bul-Python
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[1/5] Sanal ortam olusturuluyor (.venv)..." -ForegroundColor Yellow
    if ($pythonKomutu.Count -eq 2) {
        & $pythonKomutu[0] $pythonKomutu[1] -m venv (Join-Path $PSScriptRoot ".venv")
    }
    else {
        & $pythonKomutu[0] -m venv (Join-Path $PSScriptRoot ".venv")
    }
}
else {
    Write-Host "[1/5] .venv zaten mevcut, yeniden kullaniliyor." -ForegroundColor Green
}

Write-Host "[2/5] pip guncelleniyor..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[3/5] Gerekli kutuphaneler kuruluyor..." -ForegroundColor Yellow
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "[4/5] Kutuphaneler kontrol ediliyor..." -ForegroundColor Yellow
& $venvPython (Join-Path $PSScriptRoot "kurulumu_kontrol_et.py")

Write-Host "[5/5] VS Code icin Python yolu hazirlandi." -ForegroundColor Yellow
Write-Host ""
Write-Host "Kurulum tamamlandi." -ForegroundColor Green
Write-Host ""
Write-Host "VS Code Python yorumlayicisi:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\python.exe"
Write-Host ""
Write-Host "Duman testini calistirmak icin:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe calistir_duman_testi.py"
Write-Host ""
