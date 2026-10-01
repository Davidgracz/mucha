param(
    [int]$EveryHours = 6,
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackupScript = Join-Path $ProjectRoot "tools\backup_state_to_drive.ps1"
$TaskName = "Mucha Drive Backup"

if (-not (Test-Path -LiteralPath $BackupScript)) {
    throw "Brak skryptu backupu: $BackupScript"
}

$EveryHours = [Math]::Max(1, [Math]::Min(24, $EveryHours))

if ($Destination) {
    [Environment]::SetEnvironmentVariable(
        "MUCHA_BACKUP_DRIVE_DIR",
        $Destination,
        "User"
    )
    $env:MUCHA_BACKUP_DRIVE_DIR = $Destination
}

$TaskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$BackupScript`""

& schtasks.exe /Create `
    /TN $TaskName `
    /TR $TaskCommand `
    /SC HOURLY `
    /MO $EveryHours `
    /F | Out-Host

if ($LASTEXITCODE -ne 0) {
    throw "Nie udało się utworzyć zadania Harmonogramu zadań."
}

Write-Host ""
Write-Host "Utworzono: $TaskName"
Write-Host "Interwał: co $EveryHours h"
if ($Destination) {
    Write-Host "Drive: $Destination"
} else {
    Write-Host "Drive: autodetekcja / MUCHA_BACKUP_DRIVE_DIR"
}
Write-Host ""
Write-Host "Uruchamiam pierwszy backup testowy..."

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BackupScript
if ($LASTEXITCODE -ne 0) {
    throw "Zadanie zostało utworzone, ale pierwszy backup testowy nie przeszedł."
}

Write-Host ""
Write-Host "Backup działa. Zadanie będzie wykonywane automatycznie co $EveryHours h."
