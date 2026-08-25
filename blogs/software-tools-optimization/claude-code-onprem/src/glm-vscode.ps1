<#
  glm-vscode.ps1 - reversible toggle for Claude Code in VS Code between your
  normal setup and the self-hosted GLM-5.2 router.

  Claude Code shows up in VS Code two ways and they read config from two
  different places, so this flips both:

  1. terminal.integrated.env.linux in the VS Code user settings. This is what
     `claude` picks up when the extension runs it in the WSL integrated
     terminal. ON also points that at an isolated CLAUDE_CONFIG_DIR so your
     original ~/.claude on the WSL side is left alone.
  2. the env block in %USERPROFILE%\.claude\settings.json. The sidebar runs
     claude.exe on the Windows extension host, and that binary reads this file,
     not the WSL terminal env. If only the terminal block is flipped the sidebar
     stays on whatever Windows points at (for a lot of us that is a corp
     Anthropic gateway via ANTHROPIC_CUSTOM_HEADERS). So ON writes the routing
     keys here and empties ANTHROPIC_CUSTOM_HEADERS to drop the inherited
     subscription header. Your other settings in the file are kept.

  Every write is preceded by a full timestamped backup of the file it touches.
  OFF restores the most recent backup of each, byte-for-byte.

  Usage (glm-setup passes these for you):
    powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 status
    powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 on  -Port 4000 -Model glm-5.2-fp8 -Key sk-glm-local -Root /home/you/.glm-selfservice
    powershell -ExecutionPolicy Bypass -File glm-vscode.ps1 off

  After on or off, fully restart VS Code (quit all windows, a reload is not
  enough) so both the terminal and the sidebar re-read their config, and make
  sure the tunnel is up and the node router is running.

  Author: adlashab <adlashab@amd.com>
#>
param(
  [Parameter(Position=0)][ValidateSet('status','on','off')][string]$Mode='status',
  [int]$Port=4000,
  [string]$Model='glm-5.2-fp8',
  [string]$Key='sk-glm-local',
  [string]$Root='',
  [string]$Ca='',
  [string]$SettingsPath='',
  [string]$BackupDir='',
  [string]$ClaudeSettingsPath='',
  [string]$ClaudeBackupDir=''
)

$ErrorActionPreference = 'Stop'

if (-not $SettingsPath)       { $SettingsPath       = Join-Path $env:APPDATA 'Code\User\settings.json' }
if (-not $BackupDir)          { $BackupDir          = Join-Path $env:USERPROFILE '.glm-selfservice\vscode-backups' }
if (-not $ClaudeSettingsPath) { $ClaudeSettingsPath = Join-Path $env:USERPROFILE '.claude\settings.json' }
if (-not $ClaudeBackupDir)    { $ClaudeBackupDir    = Join-Path $env:USERPROFILE '.glm-selfservice\claude-backups' }
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
New-Item -ItemType Directory -Force -Path $ClaudeBackupDir | Out-Null

function Read-Json($path) { Get-Content -Raw -LiteralPath $path | ConvertFrom-Json }

# Node reads these files, so write plain UTF-8 with no BOM.
function Write-Json($obj, $path) {
  $text = ($obj | ConvertTo-Json -Depth 32)
  [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Backup-File($path, $dir) {
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $dest  = Join-Path $dir ((Split-Path -Leaf $path) + ".$stamp.bak")
  Copy-Item -LiteralPath $path -Destination $dest -Force
  Write-Host "backed up -> $dest"
}

function Restore-Latest($path, $dir, $label) {
  $filter = (Split-Path -Leaf $path) + '.*.bak'
  $bak = Get-ChildItem -LiteralPath $dir -Filter $filter -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $bak) { Write-Host "no $label backup in $dir; left as-is."; return $false }
  Copy-Item -LiteralPath $bak.FullName -Destination $path -Force
  Write-Host "restored $label from $($bak.FullName)"
  return $true
}

function Resolve-Root {
  if ($Root) { return $Root }
  try { $h = (& wsl.exe -e bash -lc 'printf %s "$HOME"') } catch { $h = $null }
  if ($h) { return "$h/.glm-selfservice" }
  throw "Could not determine your WSL install path. Pass -Root /home/<you>/.glm-selfservice"
}

# terminal.integrated.env.linux: what `claude` reads in the WSL terminal.
function Set-TerminalEnv {
  $r = Resolve-Root
  if (-not (Test-Path $SettingsPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SettingsPath) | Out-Null
    Write-Json ([pscustomobject]@{}) $SettingsPath
  }
  Backup-File $SettingsPath $BackupDir
  $env_block = [ordered]@{
    'CLAUDE_CONFIG_DIR'                        = "$r/config"
    'PATH'                                     = "$r/node/bin:$r/npm-global/bin:`${env:PATH}"
    'ANTHROPIC_BASE_URL'                       = "http://127.0.0.1:$Port"
    'ANTHROPIC_API_KEY'                        = $Key
    'ANTHROPIC_AUTH_TOKEN'                     = $Key
    'ANTHROPIC_MODEL'                          = $Model
    'ANTHROPIC_SMALL_FAST_MODEL'               = $Model
    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC' = '1'
  }
  if ($Ca) {
    $env_block['NODE_EXTRA_CA_CERTS'] = $Ca
    $env_block['CURL_CA_BUNDLE']      = $Ca
  }
  $j = Read-Json $SettingsPath
  $obj = [ordered]@{}; foreach ($k in $env_block.Keys) { $obj[$k] = $env_block[$k] }
  $j.PSObject.Properties.Remove('terminal.integrated.env.linux')
  $j | Add-Member -NotePropertyName 'terminal.integrated.env.linux' -NotePropertyValue ([pscustomobject]$obj)
  Write-Json $j $SettingsPath
  Write-Host "VS Code linux terminal -> GLM-5.2 router on http://127.0.0.1:$Port"
}

# %USERPROFILE%\.claude\settings.json env: what the Windows sidebar's claude.exe
# reads. We merge our keys in and keep any other settings in the file.
function Set-ClaudeEnv {
  if (-not (Test-Path $ClaudeSettingsPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ClaudeSettingsPath) | Out-Null
    Write-Json ([pscustomobject]@{}) $ClaudeSettingsPath
  }
  Backup-File $ClaudeSettingsPath $ClaudeBackupDir
  $desired = [ordered]@{
    'ANTHROPIC_BASE_URL'                       = "http://127.0.0.1:$Port"
    'ANTHROPIC_API_KEY'                        = $Key
    'ANTHROPIC_AUTH_TOKEN'                     = $Key
    'ANTHROPIC_MODEL'                          = $Model
    'ANTHROPIC_SMALL_FAST_MODEL'               = $Model
    'ANTHROPIC_CUSTOM_HEADERS'                 = ''
    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC' = '1'
  }
  $j = Read-Json $ClaudeSettingsPath
  $merged = [ordered]@{}
  if ($j.PSObject.Properties['env'] -and ($j.env -is [System.Management.Automation.PSCustomObject])) {
    foreach ($p in $j.env.PSObject.Properties) { $merged[$p.Name] = $p.Value }
  }
  foreach ($k in $desired.Keys) { $merged[$k] = $desired[$k] }
  $j.PSObject.Properties.Remove('env')
  $j | Add-Member -NotePropertyName 'env' -NotePropertyValue ([pscustomobject]$merged)
  Write-Json $j $ClaudeSettingsPath
  Write-Host "Claude sidebar ($ClaudeSettingsPath) -> GLM-5.2 router on http://127.0.0.1:$Port"
}

function Show-State($path, $envValue, $label) {
  if (-not (Test-Path $path)) { Write-Host "${label}: no settings at $path"; return }
  if ($envValue -and $envValue.ANTHROPIC_BASE_URL) {
    Write-Host "$label ANTHROPIC_BASE_URL = $($envValue.ANTHROPIC_BASE_URL)"
    if ($envValue.ANTHROPIC_BASE_URL -like '*127.0.0.1*' -or $envValue.ANTHROPIC_BASE_URL -like '*localhost*') {
      Write-Host '  state: GLM-5.2 (local router tunnel)'
    } else {
      Write-Host '  state: not the GLM router'
    }
  } else {
    Write-Host "${label}: no ANTHROPIC_BASE_URL set"
  }
}

switch ($Mode) {

  'status' {
    $lin = if (Test-Path $SettingsPath) { (Read-Json $SettingsPath).'terminal.integrated.env.linux' } else { $null }
    Show-State $SettingsPath $lin 'linux terminal'
    $cenv = if (Test-Path $ClaudeSettingsPath) { (Read-Json $ClaudeSettingsPath).env } else { $null }
    Show-State $ClaudeSettingsPath $cenv 'claude sidebar'
    Write-Host "vscode settings: $SettingsPath"
    Write-Host "vscode backups:  $BackupDir"
    Write-Host "claude settings: $ClaudeSettingsPath"
    Write-Host "claude backups:  $ClaudeBackupDir"
  }

  'on' {
    Set-TerminalEnv
    Set-ClaudeEnv
    # both surfaces are load-bearing (terminal + sidebar), so read them back and
    # fail loudly if either one didn't actually take. A silent no-op here is what
    # left the sidebar on the old endpoint before.
    $lin  = (Read-Json $SettingsPath).'terminal.integrated.env.linux'
    $cenv = (Read-Json $ClaudeSettingsPath).env
    if (-not ($lin  -and "$($lin.ANTHROPIC_BASE_URL)"  -like '*127.0.0.1*')) {
      throw "terminal env in $SettingsPath did not get the GLM base url"
    }
    if (-not ($cenv -and "$($cenv.ANTHROPIC_BASE_URL)" -like '*127.0.0.1*')) {
      throw "sidebar env in $ClaudeSettingsPath did not get the GLM base url"
    }
    Write-Host "both surfaces flipped: WSL terminal and Windows sidebar -> GLM-5.2 router on http://127.0.0.1:$Port"
    Write-Host 'Next: fully restart VS Code (quit all windows), bring up the tunnel (glm-code --check), ensure the node router is up.'
  }

  'off' {
    $a = Restore-Latest $SettingsPath       $BackupDir       'VS Code settings'
    $b = Restore-Latest $ClaudeSettingsPath $ClaudeBackupDir 'Claude settings'
    if (-not $a -and -not $b) { throw "no backups found in $BackupDir or $ClaudeBackupDir; restore your settings manually." }
    Write-Host 'Next: fully restart VS Code (quit all windows).'
  }
}
