$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($env:MODEL_PATH)) {
  $env:MODEL_PATH = Join-Path $repoRoot 'runs/laya-typed-decisions-zh'
} elseif (-not [System.IO.Path]::IsPathRooted($env:MODEL_PATH)) {
  $env:MODEL_PATH = Join-Path $repoRoot $env:MODEL_PATH
}

if (-not (Test-Path (Join-Path $env:MODEL_PATH 'model.safetensors'))) {
  throw "Checkpoint not found at '$env:MODEL_PATH'. Set MODEL_PATH to the training output directory."
}

if ([string]::IsNullOrWhiteSpace($env:DEVICE)) { $env:DEVICE = 'cuda' }
$env:MODEL_NAME = if ([string]::IsNullOrWhiteSpace($env:MODEL_NAME)) { 'laya-typed-decisions-zh' } else { $env:MODEL_NAME }
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

Push-Location $repoRoot
$backendExit = 0
try {
  conda run --no-capture-output -n base python demo-app/run-backend.py
  $backendExit = $LASTEXITCODE
} finally {
  Pop-Location
}
if ($backendExit -ne 0) { exit $backendExit }
