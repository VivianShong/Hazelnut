<#
.SYNOPSIS
    Install the /self-improve slash command for GitHub Copilot.

.DESCRIPTION
    Installs the self-improve workflow into the locations Copilot reads custom
    commands from, so "/self-improve" becomes available:

      * VS Code user prompts folder      (Copilot Chat, all workspaces)
        <- .github/prompts/self-improve.prompt.md
      * ~/.copilot/skills/self-improve   (GitHub Copilot CLI)
        <- .github/skills/self-improve/SKILL.md

    NOTE: GitHub Copilot CLI does NOT read prompt files (~/.copilot/prompts is
    ignored). Its invocable-workflow primitive is a *skill* (SKILL.md), so the
    CLI target installs the skill, not the prompt. Both also live in the repo
    under .github/, which makes them available automatically inside this repo.

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

# Ensure uv (Python project manager) is installed first.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # Make uv available in the current session without a restart.
    $uvBin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path $uvBin) {
        $env:Path = "$uvBin;$env:Path"
    }
}
else {
    Write-Host "uv already installed: $((Get-Command uv).Source)"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PromptSource = Join-Path $RepoRoot '.github/prompts/self-improve.prompt.md'
$SkillSource = Join-Path $RepoRoot '.github/skills/self-improve/SKILL.md'

if (-not (Test-Path $PromptSource)) {
    throw "Prompt file not found at: $PromptSource"
}
if (-not (Test-Path $SkillSource)) {
    throw "Skill file not found at: $SkillSource"
}

if ($Targets -in @('vscode', 'all')) {
    # VS Code Copilot Chat reads *.prompt.md slash commands from the user
    # prompts folder (Stable on Windows).
    $vscodePrompts = Join-Path $env:APPDATA 'Code/User/prompts'
    if (-not (Test-Path $vscodePrompts)) {
        New-Item -ItemType Directory -Path $vscodePrompts -Force | Out-Null
    }
    $target = Join-Path $vscodePrompts 'self-improve.prompt.md'
    Copy-Item -Path $PromptSource -Destination $target -Force
    Write-Host "Installed /self-improve (VS Code prompt) -> $target"
}

if ($Targets -in @('cli', 'all')) {
    # GitHub Copilot CLI reads skills from ~/.copilot/skills/<name>/SKILL.md.
    # (It does NOT read ~/.copilot/prompts.) Copy the whole skill directory so
    # the bundled tools/ scripts ship alongside SKILL.md.
    $skillSrcDir = Split-Path -Parent $SkillSource
    $cliSkillDir = Join-Path $HOME '.copilot/skills/self-improve'
    if (-not (Test-Path $cliSkillDir)) {
        New-Item -ItemType Directory -Path $cliSkillDir -Force | Out-Null
    }
    Copy-Item -Path (Join-Path $skillSrcDir '*') -Destination $cliSkillDir -Recurse -Force
    Write-Host "Installed /self-improve (CLI skill)   -> $(Join-Path $cliSkillDir 'SKILL.md')"
}

Write-Host ""
Write-Host "Done. Use '/self-improve' in VS Code Copilot Chat, or in the Copilot CLI"
Write-Host "prompt it like: 'Use the /self-improve skill to start the training loop.'"
Write-Host "In an existing CLI session, run '/skills reload' then '/skills info self-improve'."
Write-Host "Note: install deps first with 'uv sync'."
