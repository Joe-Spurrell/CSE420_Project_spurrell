param(
    [string]$Speeds = '1,2,3,4,5',
    [int]$Steps = 20
)
$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$benchmark = Join-Path $projectRoot 'benchmark_environment.py'
$blocks = Join-Path $projectRoot 'environment\Easy\Blocks.exe'
$settingsFolder = Join-Path $projectRoot 'environment\clock_test'
$settingsLoadFolder = Join-Path $projectRoot 'environment\Easy\Custom\Binaries\Win64'
$activeSettingsPath = Join-Path $settingsLoadFolder 'settings.json'
$supportedSpeeds = @(1, 2, 3, 4, 5, 10, 20, 30)
$selectedSpeeds = @($Speeds -split ',' | ForEach-Object { [int]$_.Trim() })

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $blocks)) { throw "AirSim executable not found: $blocks" }
if (-not (Test-Path -LiteralPath $settingsLoadFolder)) { throw "AirSim executable settings folder not found: $settingsLoadFolder" }
if ($Steps -le 0) { throw '-Steps must be a positive integer.' }
foreach ($clock in $selectedSpeeds) {
    if ($clock -notin $supportedSpeeds) {
        throw "Unsupported speed ${clock}x. Choose from: $($supportedSpeeds -join ', ')."
    }
}

$listenerCount = @(Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue).Count
if ($listenerCount -gt 0) {
    throw 'AirSim is already listening on port 41451. Close the existing Blocks simulator before benchmarking so each speed is loaded at startup.'
}
New-Item -ItemType Directory -Force -Path $settingsFolder | Out-Null

$previousClockSpeed = $env:AIRSIM_CLOCK_SPEED
$previousSettingsPath = $env:AIRSIM_SETTINGS_PATH
$activeSettingsExisted = Test-Path -LiteralPath $activeSettingsPath
$previousActiveSettings = if ($activeSettingsExisted) { Get-Content -LiteralPath $activeSettingsPath -Raw } else { $null }
try {
    foreach ($clock in $selectedSpeeds) {
        $settingsPath = Join-Path $settingsFolder "settings-$clock.json"

        $settingsJson = [ordered]@{
            SettingsVersion = 1.2
            SimMode = 'Multirotor'
            ViewMode = 'NoDisplay'
            ClockType = 'ScalableClock'
            ClockSpeed = $clock
        } | ConvertTo-Json
        Set-Content -LiteralPath $settingsPath -Value $settingsJson -Encoding Ascii
        # This packaged AirSim build searches next to its nested Unreal executable
        # before the user Documents folder; keep the active file local to the repo.
        Set-Content -LiteralPath $activeSettingsPath -Value $settingsJson -Encoding Ascii

        $env:AIRSIM_CLOCK_SPEED = [string]$clock
        $env:AIRSIM_SETTINGS_PATH = $settingsPath
        Write-Host "`n===== Testing ClockSpeed ${clock}x =====" -ForegroundColor Cyan

        $sim = $null
        try {
            $arguments = @('-windowed', '-ResX=1280', '-ResY=720', "-settings=$activeSettingsPath")
            # AirSim settings must be available from the simulator's startup directory too.
            $sim = Start-Process -FilePath $blocks -WorkingDirectory $settingsFolder -ArgumentList $arguments -PassThru
            & $python $benchmark --steps $Steps --clock-speed $clock
            if ($LASTEXITCODE -ne 0) { throw "Benchmark failed at ${clock}x (exit $LASTEXITCODE)." }
        }
        finally {
            if ($sim) {
                $simProcess = Get-Process -Id $sim.Id -ErrorAction SilentlyContinue
                if ($simProcess) {
                    & "$env:SystemRoot\System32\taskkill.exe" /PID $sim.Id /T /F | Out-Null
                    $sim.WaitForExit(10000) | Out-Null
                }
            }
            $closeDeadline = (Get-Date).AddSeconds(15)
            do {
                $listenerCount = @(Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue).Count
                if ($listenerCount -eq 0) { break }
                Start-Sleep -Milliseconds 500
            } while ((Get-Date) -lt $closeDeadline)
            if ($listenerCount -gt 0) {
                throw "AirSim did not release port 41451 after the ${clock}x run. Stop the leftover Blocks process before continuing."
            }
            Start-Sleep -Milliseconds 500
        }
    }
}
finally {
    if ($activeSettingsExisted) {
        Set-Content -LiteralPath $activeSettingsPath -Value $previousActiveSettings -Encoding Ascii
    } else {
        Remove-Item -LiteralPath $activeSettingsPath -Force -ErrorAction SilentlyContinue
    }
    if ($null -eq $previousClockSpeed) {
        Remove-Item Env:AIRSIM_CLOCK_SPEED -ErrorAction SilentlyContinue
    } else {
        $env:AIRSIM_CLOCK_SPEED = $previousClockSpeed
    }
    if ($null -eq $previousSettingsPath) {
        Remove-Item Env:AIRSIM_SETTINGS_PATH -ErrorAction SilentlyContinue
    } else {
        $env:AIRSIM_SETTINGS_PATH = $previousSettingsPath
    }
}

