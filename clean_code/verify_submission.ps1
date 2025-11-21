# Final Pre-Submission Verification Script (PowerShell)
# Run this before creating the submission archive

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "FINAL PRE-SUBMISSION VERIFICATION" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if in clean_code directory
if (!(Test-Path "launch.py")) {
    Write-Host "❌ Error: Must run from clean_code/ directory" -ForegroundColor Red
    exit 1
}

Write-Host "✓ In correct directory" -ForegroundColor Green
Write-Host ""

# Count files
$pythonFiles = (Get-ChildItem -Recurse -Filter "*.py").Count
$totalFiles = (Get-ChildItem -Recurse -File).Count
Write-Host "📊 Statistics:" -ForegroundColor Yellow
Write-Host "   Python files: $pythonFiles"
Write-Host "   Total files: $totalFiles"
Write-Host ""

# Check for unwanted files
Write-Host "🔍 Checking for unwanted files..." -ForegroundColor Yellow
$pycache = Get-ChildItem -Recurse -Directory -Filter "__pycache__"
$pyc = Get-ChildItem -Recurse -Filter "*.pyc"
$logs = Get-ChildItem -Recurse -Filter "*.log"

if ($pycache.Count -gt 0) {
    Write-Host "   ⚠ Found $($pycache.Count) __pycache__ directories - cleaning..." -ForegroundColor Yellow
    $pycache | Remove-Item -Recurse -Force
} else {
    Write-Host "   ✓ No __pycache__ directories" -ForegroundColor Green
}

if ($pyc.Count -gt 0) {
    Write-Host "   ⚠ Found $($pyc.Count) .pyc files - cleaning..." -ForegroundColor Yellow
    $pyc | Remove-Item -Force
} else {
    Write-Host "   ✓ No .pyc files" -ForegroundColor Green
}

if ($logs.Count -gt 0) {
    Write-Host "   ⚠ Found $($logs.Count) .log files" -ForegroundColor Yellow
} else {
    Write-Host "   ✓ No .log files" -ForegroundColor Green
}
Write-Host ""

# Check key files
Write-Host "📄 Checking key files..." -ForegroundColor Yellow
$keyFiles = @("launch.py", "requirements.txt", "README.md", "QUICK_START.md", "SUBMISSION_CHECKLIST.md")
foreach ($file in $keyFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file MISSING" -ForegroundColor Red
    }
}
Write-Host ""

# Check directory structure
Write-Host "📁 Checking directory structure..." -ForegroundColor Yellow
$keyDirs = @("core/adapters", "core/infrastructure", "core/modules", "tests", "configs", "models", "test_results")
foreach ($dir in $keyDirs) {
    if (Test-Path $dir) {
        Write-Host "   ✓ $dir/" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $dir/ MISSING" -ForegroundColor Red
    }
}
Write-Host ""

# Check for sensitive data (basic check)
Write-Host "🔒 Checking for sensitive data..." -ForegroundColor Yellow
try {
    $apiKeyPattern = "MISTRAL_API_KEY.*=.*`"[a-zA-Z0-9]"
    $sensitiveFiles = Select-String -Path "*.py" -Pattern $apiKeyPattern -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Line -notmatch "environment|environ|getenv" }
    if ($sensitiveFiles) {
        Write-Host "   ⚠ WARNING: Possible hardcoded API key found!" -ForegroundColor Yellow
        $sensitiveFiles | ForEach-Object { Write-Host "     $($_.Filename):$($_.LineNumber)" -ForegroundColor Gray }
    } else {
        Write-Host "   ✓ No hardcoded API keys detected" -ForegroundColor Green
    }

    $passwordPattern = "password.*=.*`""
    $passwordFiles = Select-String -Path "*.py" -Pattern $passwordPattern -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Line -notmatch "#" }
    if ($passwordFiles) {
        Write-Host "   ⚠ WARNING: Possible hardcoded password found!" -ForegroundColor Yellow
    } else {
        Write-Host "   ✓ No hardcoded passwords detected" -ForegroundColor Green
    }
} catch {
    Write-Host "   ⚠ Could not complete sensitive data check" -ForegroundColor Yellow
}
Write-Host ""

# Calculate size
Write-Host "📦 Package size:" -ForegroundColor Yellow
$totalSize = [math]::Round((Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum/1MB, 2)
Write-Host "   Total: $totalSize MB"
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "FINAL CHECKLIST:" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Before submission, ensure:"
Write-Host "  [ ] All personal information removed"
Write-Host "  [ ] No hardcoded API keys or credentials"
Write-Host "  [ ] README updated with your information"
Write-Host "  [ ] Citation section filled out"
Write-Host "  [ ] Test scripts verified to work"
Write-Host "  [ ] MISTRAL_API_KEY documented in README"
Write-Host ""
Write-Host "To create submission archive:" -ForegroundColor Yellow
Write-Host "  cd .."
Write-Host "  Compress-Archive -Path clean_code -DestinationPath lumenaa_submission.zip"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Verification Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
