# Create Desktop Shortcut for Emotion Recommender
# Run this script as administrator to create a desktop shortcut

# Clear screen and show header
Clear-Host
Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "   Create Desktop Shortcut for Emotion Recommender" -ForegroundColor Green
Write-Host "===============================================`n" -ForegroundColor Cyan

# Get the current script directory (project root)
$projectPath = $PSScriptRoot
$launcherPath = Join-Path -Path $projectPath -ChildPath "Start_Application.ps1"

# Verify that the launcher script exists
if (-not (Test-Path $launcherPath)) {
    Write-Host "❌ ERROR: Launcher script not found at: $launcherPath" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    Write-Host "`nPress any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Get desktop path
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path -Path $desktopPath -ChildPath "Emotion Recommender.lnk"

# Create shortcut
try {
    Write-Host "Creating shortcut on desktop..." -ForegroundColor Yellow
    
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($shortcutPath)
    $Shortcut.TargetPath = "powershell.exe"
    $Shortcut.Arguments = "-ExecutionPolicy Bypass -NoProfile -File `"$launcherPath`""
    $Shortcut.WorkingDirectory = $projectPath
    $Shortcut.IconLocation = "powershell.exe,0"
    $Shortcut.Description = "Launch Emotion-Aware Intelligent Video Recommendation System"
    $Shortcut.Save()
    
    Write-Host "`n✅ Shortcut created successfully at: $shortcutPath" -ForegroundColor Green
}
catch {
    Write-Host "`n❌ ERROR: Failed to create shortcut" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nYou can now launch the application by double-clicking the shortcut on your desktop.`n" -ForegroundColor Cyan
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
