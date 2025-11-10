#!/bin/bash
# Script to prepare align-rerank project for Google Colab upload
# This creates a clean zip file with all the updated code

set -e

echo "📦 Preparing align-rerank for Colab upload..."

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PROJECT_NAME="align-rerank"

# Create zip file name with timestamp
ZIP_NAME="../${PROJECT_NAME}-$(date +%Y%m%d-%H%M%S).zip"
# Or use a simple name
ZIP_NAME_SIMPLE="../${PROJECT_NAME}-updated.zip"

echo "📁 Project directory: $PROJECT_DIR"
echo "📦 Creating zip file: $ZIP_NAME_SIMPLE"

# Create zip excluding unnecessary files
zip -r "$ZIP_NAME_SIMPLE" . \
    -x "*.git*" \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x "*.ipynb_checkpoints/*" \
    -x ".DS_Store" \
    -x "*.swp" \
    -x "*.swo" \
    -x "runs/*" \
    -x "results/*" \
    -x ".venv/*" \
    -x "*.zip"

echo ""
echo "✅ Zip file created: $ZIP_NAME_SIMPLE"
echo ""
echo "📋 File contents:"
unzip -l "$ZIP_NAME_SIMPLE" | head -30
echo ""
echo "🚀 Ready to upload to Colab!"
echo "   Size: $(du -h "$ZIP_NAME_SIMPLE" | cut -f1)"

