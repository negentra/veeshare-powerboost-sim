param(
    [int[]]$Seeds = @(100, 200, 300, 400),
    [string]$RunLabel = "final",
    [ValidateSet("audit", "full")]
    [string]$Mode = "full"
)

# Runs the audit pipeline once per seed using run.py's --seed override.
# config/seed.yaml is NOT modified (unlike earlier versions); the effective
# seed is passed on the command line and recorded in each run_manifest.json.

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$runRoot = Join-Path $root "outputs_multiseed_$RunLabel"
$logRoot = Join-Path $runRoot "_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"

foreach ($seed in $Seeds) {
    $outDir = Join-Path $runRoot "seed$seed"
    $stdout = Join-Path $logRoot "seed${seed}.stdout.log"
    $stderr = Join-Path $logRoot "seed${seed}.stderr.log"

    "=== seed $seed started $(Get-Date -Format o), mode=$Mode ===" |
        Tee-Object -FilePath $stdout

    & python run.py --mode $Mode --seed $seed --out-dir $outDir 1>> $stdout 2>> $stderr
    $exitCode = $LASTEXITCODE

    "=== seed $seed finished $(Get-Date -Format o), exit=$exitCode ===" |
        Tee-Object -FilePath $stdout -Append

    if ($exitCode -ne 0) {
        throw "seed $seed failed with exit code $exitCode. See $stderr"
    }
}

"DONE $runRoot"
