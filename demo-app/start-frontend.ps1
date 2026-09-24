$ErrorActionPreference = 'Stop'
$frontendRoot = Join-Path $PSScriptRoot 'frontend'
Push-Location $frontendRoot
try {
  if (-not (Test-Path 'node_modules')) { npm ci }
  npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
} finally {
  Pop-Location
}
