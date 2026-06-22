<#
.SYNOPSIS
    Install the /self-improve slash command for GitHub Copilot.

.DESCRIPTION
    Copies grpo's self-improve prompt file into the locations Copilot reads
    custom slash commands from, so "/self-improve" becomes available:

      * VS Code user prompts folder  (Copilot Chat, all workspaces)
      * ~/.copilot/prompts           (GitHub Copilot CLI)

    The prompt also lives in the repo at .github/prompts/self-improve.prompt.md,
    which makes it available automatically when working inside this repository.

.PARAMETER Targets
    Which install locations to write to: 'vscode', 'cli', or 'all' (default).

.EXAMPLE
    ./scripts/install_self_improve_skill.ps1
    ./scripts/install_self_improve_skill.ps1 -Targets cli
#>

[CmdletBinding()]
param(
    [ValidateSet('vscode', 'cli', 'all')]
    [string]$Targets = 'all'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $RepoRoot '.github/prompts/self-improve.prompt.md'

if (-not (Test-Path $Source)) {
    throw "Prompt file not found at: $Source"
}

# Resolve destination prompt directories.
$dests = @()

if ($Targets -in @('vscode', 'all')) {
    # VS Code user prompts folder (Stable on Windows).
    $vscodePrompts = Join-Path $env:APPDATA 'Code/User/prompts'
    $dests += $vscodePrompts
}

if ($Targets -in @('cli', 'all')) {
    # GitHub Copilot CLI custom prompts directory.
    $cliPrompts = Join-Path $HOME '.copilot/prompts'
    $dests += $cliPrompts
}

foreach ($dir in $dests) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $target = Join-Path $dir 'self-improve.prompt.md'
    Copy-Item -Path $Source -Destination $target -Force
    Write-Host "Installed /self-improve -> $target"
}

Write-Host ""
Write-Host "Done. Type '/self-improve' in Copilot Chat or the Copilot CLI to run a"
Write-Host "training generation and switch to the fine-tuned model."
Write-Host "Note: install deps first with 'uv sync --extra grpo'."
