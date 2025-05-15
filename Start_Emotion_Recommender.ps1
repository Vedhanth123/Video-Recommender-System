# Emotion-Aware Intelligent Video Recommendation Framework Launcher
# PowerShell Script for easy application startup

# Clear the screen and display a welcome message
Clear-Host
Write-Host "`n`n=============================================================" -ForegroundColor Cyan
Write-Host "    Emotion-Aware Intelligent Video Recommendation System" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "`nStarting application..." -ForegroundColor Yellow

# Store the project path - this will be changed to the desktop shortcut path
$projectPath = "c:\Code\Major-Project"

# Navigate to the project directory
try {
    Set-Location -Path $projectPath -ErrorAction Stop
    Write-Host "✓ Project folder found" -ForegroundColor Green
}
catch {
    Write-Host "✗ ERROR: Could not locate project folder at $projectPath" -ForegroundColor Red
    Write-Host "Please ensure the path in this script is correct." -ForegroundColor Red
    Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Activate the virtual environment
Write-Host "`nActivating Python virtual environment..." -ForegroundColor Yellow
$venvActivatePath = Join-Path -Path $projectPath -ChildPath "MP\Scripts\Activate.ps1"

try {
    if (Test-Path $venvActivatePath) {
        & $venvActivatePath
        Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    }
    else {
        throw "Virtual environment activation script not found"
    }
}
catch {
    Write-Host "✗ ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Set environment variables to suppress warnings
Write-Host "`nConfiguring environment settings..." -ForegroundColor Yellow
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
Write-Host "✓ Environment variables set" -ForegroundColor Green

# Launch the application using the launcher script or directly with Streamlit
Write-Host "`nLaunching application..." -ForegroundColor Green
try {
    # Check if launcher.py exists, use it if available
    if (Test-Path "launcher.py") {
        Write-Host "Starting with launcher.py" -ForegroundColor Cyan
        python launcher.py
    }
    else {
        # Fallback to direct Streamlit launch
        Write-Host "Starting with Streamlit directly" -ForegroundColor Cyan
        streamlit run app.py --server.headless true --browser.serverAddress localhost --browser.gatherUsageStats false --theme.base light
    }
    
    Write-Host "`n✓ Application closed successfully" -ForegroundColor Green
}
catch {
    Write-Host "✗ ERROR: Failed to start the application" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
}

# Pause before exiting
Write-Host "`nThank you for using the Emotion-Aware Intelligent Video Recommendation System!" -ForegroundColor Cyan
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
