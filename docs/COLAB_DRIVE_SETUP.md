# Colab Setup - Using Google Drive

If you uploaded the zip file to Google Drive, use this code instead:

## Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

This will prompt you to authenticate. Click the link, authorize, and paste the code.

## Step 2: Load and Extract from Drive

```python
import zipfile
import os
import sys

# Path to your zip file on Drive (adjust if needed)
# Usually it's in: MyDrive/align-rerank-updated.zip
ZIP_PATH = '/content/drive/MyDrive/align-rerank-updated.zip'

# Alternative: find it automatically
# drive_path = '/content/drive/MyDrive'
# zip_files = [f for f in os.listdir(drive_path) if f.endswith('.zip') and 'align-rerank' in f]
# if zip_files:
#     ZIP_PATH = os.path.join(drive_path, zip_files[0])
#     print(f"Found: {ZIP_PATH}")

# Check if file exists
if os.path.exists(ZIP_PATH):
    print(f"âœ… Found zip file: {ZIP_PATH}")
    
    # Extract it
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    print(f"âœ… Extracted align-rerank project")
    
    # Verify extraction
    if os.path.exists('align-rerank'):
        print("âœ… align-rerank folder created")
        
        # Check for new files
        if os.path.exists('align-rerank/src/align_rerank/score_factcc.py'):
            print("âœ… score_factcc.py (NEW) - Found!")
        if os.path.exists('align-rerank/src/align_rerank/create_baseline.py'):
            print("âœ… create_baseline.py (NEW) - Found!")
        if os.path.exists('align-rerank/src/align_rerank/verifiers/qags.py'):
            print("âœ… qags.py (NEW) - Found!")
        
        # Add to Python path
        sys.path.insert(0, 'align-rerank/src')
        print("\nâœ… Project ready! Python path updated.")
    else:
        print("âŒ Extraction failed - check zip file")
else:
    print(f"âŒ Zip file not found at: {ZIP_PATH}")
    print("   Please check the path in Google Drive")
    print("   Or use: !ls /content/drive/MyDrive/ | grep zip")
```

## Step 3: Alternative - Auto-find the zip file

If you're not sure where the file is:

```python
import os
import zipfile
import sys

# Mount drive first (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# Search for the zip file
drive_path = '/content/drive/MyDrive'
print(f"ðŸ” Searching in: {drive_path}")

# Find all zip files
zip_files = []
for root, dirs, files in os.walk(drive_path):
    for file in files:
        if file.endswith('.zip') and 'align-rerank' in file.lower():
            full_path = os.path.join(root, file)
            zip_files.append(full_path)
            print(f"   Found: {full_path}")

if zip_files:
    # Use the first one found
    ZIP_PATH = zip_files[0]
    print(f"\nâœ… Using: {ZIP_PATH}")
    
    # Extract
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    print("âœ… Extracted!")
    
    # Add to path
    if os.path.exists('align-rerank'):
        sys.path.insert(0, 'align-rerank/src')
        print("âœ… Python path updated")
else:
    print("âŒ No align-rerank zip file found")
    print("   Make sure the file is in your Google Drive")
```

