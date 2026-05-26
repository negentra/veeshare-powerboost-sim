param(
    [int[]]$Seeds = @(100, 200, 300, 400),
    [string]$RunLabel = "final",
    [ValidateSet("audit", "full")]
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$runRoot = Join-Path $root "outputs_multiseed_$RunLabel"
$logRoot = Join-Path $runRoot "_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$seedPath = Join-Path $root "config\seed.yaml"
$originalSeedYaml = Get-Content -Raw -LiteralPath $seedPath

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"

try {
    foreach ($seed in $Seeds) {
        $outDir = Join-Path $runRoot "seed$seed"
        $stdout = Join-Path $logRoot "seed${seed}.stdout.log"
        $stderr = Join-Path $logRoot "seed${seed}.stderr.log"

        @"
# seed.yaml
# Root seed for all stochastic operations in the simulation.
# Per-module seeds are derived deterministically from this via:
#   module_seed = int(sha256(root_seed.bytes + module_name.bytes).hexdigest()[:8], 16)
# See src/utils/seed.py.
# Changing this value yields a fully independent realisation; keeping it
# constant guarantees bit-exact reproduction of all 20 headline metrics.
seed_root: $seed
"@ | Set-Content -LiteralPath $seedPath -Encoding UTF8

        "=== seed $seed started $(Get-Date -Format o), mode=$Mode ===" |
            Tee-Object -FilePath $stdout

        & python run.py --mode $Mode --out-dir $outDir 1>> $stdout 2>> $stderr
        $exitCode = $LASTEXITCODE

        "=== seed $seed finished $(Get-Date -Format o), exit=$exitCode ===" |
            Tee-Object -FilePath $stdout -Append

        if ($exitCode -ne 0) {
            throw "seed $seed failed with exit code $exitCode. See $stderr"
        }
    }
} finally {
    Set-Content -LiteralPath $seedPath -Value $originalSeedYaml -Encoding UTF8
    "Restored config\seed.yaml at $(Get-Date -Format o)" |
        Tee-Object -FilePath (Join-Path $logRoot "restore.log")
}

"DONE $runRoot"
