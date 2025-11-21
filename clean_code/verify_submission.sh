#!/bin/bash
# Final Pre-Submission Verification Script
# Run this before creating the submission archive

echo "=========================================="
echo "FINAL PRE-SUBMISSION VERIFICATION"
echo "=========================================="
echo ""

# Check if in clean_code directory
if [ ! -f "launch.py" ]; then
    echo "❌ Error: Must run from clean_code/ directory"
    exit 1
fi

echo "✓ In correct directory"
echo ""

# Count files
PYTHON_FILES=$(find . -name "*.py" | wc -l)
TOTAL_FILES=$(find . -type f | wc -l)
echo "📊 Statistics:"
echo "   Python files: $PYTHON_FILES"
echo "   Total files: $TOTAL_FILES"
echo ""

# Check for unwanted files
echo "🔍 Checking for unwanted files..."
PYCACHE=$(find . -type d -name "__pycache__" | wc -l)
PYC=$(find . -name "*.pyc" | wc -l)
LOGS=$(find . -name "*.log" | wc -l)

if [ $PYCACHE -gt 0 ]; then
    echo "   ⚠ Found $PYCACHE __pycache__ directories - cleaning..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
else
    echo "   ✓ No __pycache__ directories"
fi

if [ $PYC -gt 0 ]; then
    echo "   ⚠ Found $PYC .pyc files - cleaning..."
    find . -name "*.pyc" -delete
else
    echo "   ✓ No .pyc files"
fi

if [ $LOGS -gt 0 ]; then
    echo "   ⚠ Found $LOGS .log files"
else
    echo "   ✓ No .log files"
fi
echo ""

# Check key files
echo "📄 Checking key files..."
KEY_FILES=("launch.py" "requirements.txt" "README.md" "QUICK_START.md" "SUBMISSION_CHECKLIST.md")
for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ❌ $file MISSING"
    fi
done
echo ""

# Check directory structure
echo "📁 Checking directory structure..."
KEY_DIRS=("core/adapters" "core/infrastructure" "core/modules" "tests" "configs" "models" "test_results")
for dir in "${KEY_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ❌ $dir/ MISSING"
    fi
done
echo ""

# Check for sensitive data (basic check)
echo "🔒 Checking for sensitive data..."
if grep -r "MISTRAL_API_KEY.*=.*\"[a-zA-Z0-9]" . --include="*.py" 2>/dev/null | grep -v "environment\|environ\|getenv" | grep -q .; then
    echo "   ⚠ WARNING: Possible hardcoded API key found!"
else
    echo "   ✓ No hardcoded API keys detected"
fi

if grep -r "password.*=.*\"" . --include="*.py" 2>/dev/null | grep -v "#" | grep -q .; then
    echo "   ⚠ WARNING: Possible hardcoded password found!"
else
    echo "   ✓ No hardcoded passwords detected"
fi
echo ""

# Calculate size
echo "📦 Package size:"
TOTAL_SIZE=$(du -sh . | cut -f1)
echo "   Total: $TOTAL_SIZE"
echo ""

echo "=========================================="
echo "FINAL CHECKLIST:"
echo "=========================================="
echo "Before submission, ensure:"
echo "  [ ] All personal information removed"
echo "  [ ] No hardcoded API keys or credentials"
echo "  [ ] README updated with your information"
echo "  [ ] Citation section filled out"
echo "  [ ] Test scripts verified to work"
echo "  [ ] MISTRAL_API_KEY documented in README"
echo ""
echo "To create submission archive:"
echo "  cd .."
echo "  tar -czf lumenaa_submission.tar.gz clean_code/"
echo "  # or"
echo "  zip -r lumenaa_submission.zip clean_code/"
echo ""
echo "=========================================="
echo "✅ Verification Complete!"
echo "=========================================="
