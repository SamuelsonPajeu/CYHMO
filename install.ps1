<#
.SYNOPSIS
    Instala o CYHMO num venv dedicado, dentro da propria pasta do projeto.

.DESCRIPTION
    Nunca instala no Python global. Usa uv quando disponivel e
    cai para venv + pip quando nao. Com -Pinned, reproduz o ambiente de
    referencia a partir do requirements.txt.

    Ao final baixa o backend de transcricao (whisper.cpp). Use -SkipBackend para pular.

    -FromLauncher cala as instrucoes de proximo passo: quem chamou foi o CYHMO.cmd, que
    segue sozinho para o diagnostico e para o mod. Sem isso o usuario le "rode o doctor"
    no meio de uma janela que ja esta subindo o mod, e fecha o terminal.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Pinned
    .\install.ps1 -SkipBackend
#>
[CmdletBinding()]
param(
    [switch]$Pinned,
    [switch]$SkipBackend,
    [switch]$FromLauncher,
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$MinimumPython = [Version]"3.11"

function Invoke-Native {
    <#
      No PowerShell 5.1, qualquer coisa que um .exe escreva em stderr vira erro
      quando ErrorActionPreference = Stop -- e uv e pip relatam progresso por ali.
      Quem decide sucesso aqui e o codigo de saida, nao o stderr.
    #>
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments = @(), [string]$What = "comando")
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$What falhou (codigo $LASTEXITCODE)"
    }
}

function Find-Python {
    $candidates = @(
        @{ Cmd = "py";     Launcher = @("-3.11") },
        @{ Cmd = "py";     Launcher = @("-3") },
        @{ Cmd = "python"; Launcher = @() }
    )
    $probe = "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    foreach ($candidate in $candidates) {
        $exe = Get-Command $candidate.Cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        $version = $null
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $version = (& $exe.Source @($candidate.Launcher + @("-c", $probe)) | Select-Object -First 1)
        } catch {
            $version = $null
        } finally {
            $ErrorActionPreference = $previous
        }
        if ($LASTEXITCODE -ne 0 -or -not $version) { continue }
        $parsed = $null
        if (-not [Version]::TryParse($version.Trim(), [ref]$parsed)) { continue }
        if ([Version]::new($parsed.Major, $parsed.Minor) -ge $MinimumPython) {
            return @{ Exe = $exe.Source; Args = $candidate.Launcher; Version = $version.Trim() }
        }
    }
    return $null
}

Write-Host "== CYHMO - instalacao ==" -ForegroundColor Cyan

$uv = Get-Command uv -ErrorAction SilentlyContinue
$python = Find-Python

if ($uv) {
    Write-Host "uv encontrado" -ForegroundColor Green
} elseif ($python) {
    Write-Host "Python $($python.Version) encontrado." -ForegroundColor Green
} else {
    Write-Error "Nem uv nem Python $MinimumPython+ encontrados. Instale o uv (https://docs.astral.sh/uv/) ou o Python de python.org e tente novamente."
}

$venvPython = Join-Path $root "$VenvPath\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Host "venv existe em $VenvPath;." -ForegroundColor Yellow
} elseif ($uv) {
    Write-Host "Criando venv com uv..." -ForegroundColor Cyan
    Invoke-Native -Exe $uv.Source -Arguments @("venv", "--python", "$MinimumPython", $VenvPath) -What "uv venv"
} else {
    Write-Host "Criando venv com o modulo venv..." -ForegroundColor Cyan
    Invoke-Native -Exe $python.Exe -Arguments ($python.Args + @("-m", "venv", $VenvPath)) -What "python -m venv"
}

if (-not (Test-Path $venvPython)) { Write-Error "o venv nao foi criado em $VenvPath" }

if (-not $uv) {
    $hasPip = & $venvPython -c "import importlib.util, sys; sys.stdout.write('1' if importlib.util.find_spec('pip') else '0')"
    if ($hasPip -ne "1") {
        Write-Host "venv sem pip; instalando com ensurepip..." -ForegroundColor Cyan
        Invoke-Native -Exe $venvPython -Arguments @("-m", "ensurepip", "--upgrade") -What "ensurepip"
    }
}

if ($Pinned) {
    Write-Host "Instalando versoes pinadas (requirements.txt)..." -ForegroundColor Cyan
    if ($uv) {
        Invoke-Native $uv.Source @("pip", "install", "--python", $venvPython, "-r", "requirements.txt") "uv pip install -r"
        Invoke-Native $uv.Source @("pip", "install", "--python", $venvPython, "-e", ".", "--no-deps") "uv pip install -e"
    } else {
        Invoke-Native $venvPython @("-m", "pip", "install", "-r", "requirements.txt") "pip install -r"
        Invoke-Native $venvPython @("-m", "pip", "install", "-e", ".", "--no-deps") "pip install -e"
    }
} else {
    Write-Host "Instalando o mod e suas dependencias..." -ForegroundColor Cyan
    if ($uv) {
        Invoke-Native $uv.Source @("pip", "install", "--python", $venvPython, "-e", ".") "uv pip install"
    } else {
        Invoke-Native $venvPython @("-m", "pip", "install", "--upgrade", "pip") "pip upgrade"
        Invoke-Native $venvPython @("-m", "pip", "install", "-e", ".") "pip install"
    }
}

if (-not $SkipBackend) {
    Write-Host "Baixando o backend de transcricao (whisper.cpp)..." -ForegroundColor Cyan
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $venvPython @("-m", "cyhmo", "setup")
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Erro ao configurar whisper.cpp. Utilizando faster-whisper"
        Write-Warning "tentar novamente com .\$VenvPath\Scripts\python.exe -m cyhmo setup"
    }
}

Write-Host ""
Write-Host "Instalado." -ForegroundColor Green
if (-not $FromLauncher) {
    Write-Host "Proximo passo: abra o PCSX2 com o jogo rodando e execute" -ForegroundColor Cyan
    Write-Host "  .\$VenvPath\Scripts\python.exe -m cyhmo doctor" -ForegroundColor White
    Write-Host "O modelo de comparacao de comandos e baixado no primeiro 'run'." -ForegroundColor DarkGray
}
