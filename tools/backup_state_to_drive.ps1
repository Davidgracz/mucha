param(
    [string]$Destination = "",
    [int]$Keep = 40
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogPath = Join-Path $ProjectRoot "state\drive_backup.log"

function Write-BackupLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    $dir = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Resolve-BackupDestination {
    if ($Destination) {
        return $Destination
    }

    if ($env:MUCHA_BACKUP_DRIVE_DIR) {
        return $env:MUCHA_BACKUP_DRIVE_DIR
    }

    $candidates = New-Object System.Collections.Generic.List[string]

    if ($env:USERPROFILE) {
        $candidates.Add((Join-Path $env:USERPROFILE "My Drive"))
        $candidates.Add((Join-Path $env:USERPROFILE "Mój dysk"))
        $candidates.Add((Join-Path $env:USERPROFILE "Google Drive\My Drive"))
        $candidates.Add((Join-Path $env:USERPROFILE "Google Drive\Mój dysk"))
    }

    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $candidates.Add((Join-Path $drive.Root "My Drive"))
        $candidates.Add((Join-Path $drive.Root "Mój dysk"))
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate) {
            return (Join-Path $candidate "ChatGPT\Mucha\Backups")
        }
    }

    throw @"
Nie znaleziono lokalnego folderu Google Drive.
Zainstaluj Google Drive for desktop albo ustaw raz docelową ścieżkę:
[Environment]::SetEnvironmentVariable(
  'MUCHA_BACKUP_DRIVE_DIR',
  'G:\My Drive\ChatGPT\Mucha\Backups',
  'User'
)
Potem uruchom ten skrypt ponownie.
"@
}

try {
    $BackupDestination = Resolve-BackupDestination
    New-Item -ItemType Directory -Force -Path $BackupDestination | Out-Null

    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $PythonCommand = Get-Command python -ErrorAction Stop
        $Python = $PythonCommand.Source
    }

    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $FileName = "mucha_state_$Timestamp.zip"
    $TempArchive = Join-Path ([System.IO.Path]::GetTempPath()) $FileName
    $TargetArchive = Join-Path $BackupDestination $FileName

    if (Test-Path -LiteralPath $TempArchive) {
        Remove-Item -LiteralPath $TempArchive -Force
    }

    Write-BackupLog "START -> $TargetArchive"

    & $Python (Join-Path $ProjectRoot "tools\backup_learned_state.py") --output $TempArchive
    if ($LASTEXITCODE -ne 0) {
        throw "backup_learned_state.py zakończył się kodem $LASTEXITCODE"
    }

    Copy-Item -LiteralPath $TempArchive -Destination $TargetArchive -Force

    $SourceHash = (Get-FileHash -LiteralPath $TempArchive -Algorithm SHA256).Hash
    $TargetHash = (Get-FileHash -LiteralPath $TargetArchive -Algorithm SHA256).Hash
    if ($SourceHash -ne $TargetHash) {
        throw "SHA256 kopii na Drive nie zgadza się z lokalnym archiwum."
    }

    Remove-Item -LiteralPath $TempArchive -Force -ErrorAction SilentlyContinue

    $Keep = [Math]::Max(3, $Keep)
    $OldBackups = Get-ChildItem -LiteralPath $BackupDestination -Filter "mucha_state_*.zip" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip $Keep

    foreach ($old in $OldBackups) {
        Remove-Item -LiteralPath $old.FullName -Force
    }

    $SizeMb = [Math]::Round((Get-Item -LiteralPath $TargetArchive).Length / 1MB, 2)
    Write-BackupLog "OK -> $FileName ($SizeMb MB), SHA256 $TargetHash"
    exit 0
}
catch {
    Write-BackupLog "ERROR -> $($_.Exception.Message)"
    exit 1
}
finally {
    if ($TempArchive -and (Test-Path -LiteralPath $TempArchive)) {
        Remove-Item -LiteralPath $TempArchive -Force -ErrorAction SilentlyContinue
    }
}
