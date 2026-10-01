param(
    [int]$EveryHours = 6,
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackupScript = Join-Path $ProjectRoot "tools\backup_state_to_drive.ps1"
$TaskName = "Mucha Drive Backup"

if (-not (Test-Path -LiteralPath $BackupScript)) {
    throw "Backup script not found: $BackupScript"
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

$TaskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $BackupScript + '"'

Write-Host ""
Write-Host "Creating scheduled task: $TaskName"
Write-Host "Interval: every $EveryHours hour(s)"
Write-Host ""

& schtasks.exe /Create /TN $TaskName /TR $TaskCommand /SC HOURLY /MO $EveryHours /F | Out-Host

if ($LASTEXITCODE -ne 0) {
    throw "Could not create Windows scheduled task."
}

Write-Host ""
Write-Host "Scheduled task created."
if ($Destination) {
    Write-Host "Drive destination: $Destination"
} else {
    Write-Host "Drive destination: auto-detect or MUCHA_BACKUP_DRIVE_DIR"
}

Write-Host ""
Write-Host "Running first test backup..."

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BackupScript

if ($LASTEXITCODE -ne 0) {
    throw "Scheduled task was created, but the first test backup failed."
}

Write-Host ""
Write-Host "Backup OK."
Write-Host "The task will run automatically every $EveryHours hour(s)."
