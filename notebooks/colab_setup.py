"""
Professional Colab setup script for align-rerank project.

This script handles:
1. Google Drive mounting
2. Project extraction from zip
3. Environment setup
4. Path configuration
5. Directory creation
6. Verification

Usage in Colab:
    from notebooks.colab_setup import setup_colab
    setup_colab()
"""

from pathlib import Path
from typing import Optional
import zipfile
import os
import sys


def setup_colab(
    zip_path: Optional[str] = None,
    project_name: str = "align-rerank",
    drive_base: str = "/content/drive/MyDrive"
) -> dict:
    """
    Setup align-rerank project in Google Colab.
    
    Args:
        zip_path: Full path to zip file on Drive. If None, auto-searches.
        project_name: Expected project folder name
        drive_base: Base path for Google Drive
        
    Returns:
        dict with setup status and paths
    """
    status = {
        "success": False,
        "project_path": None,
        "src_path": None,
        "error": None
    }
    
    try:
        # Step 1: Mount Google Drive
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            print("âœ… Google Drive mounted")
        except Exception as e:
            print(f"âš ï¸  Drive mount: {e}")
            # Continue anyway - might already be mounted
        
        # Step 2: Find zip file
        if zip_path is None:
            zip_path = _find_zip_file(drive_base, project_name)
        
        if not zip_path or not os.path.exists(zip_path):
            status["error"] = f"Zip file not found: {zip_path}"
            print(f"âŒ {status['error']}")
            return status
        
        print(f"ðŸ“¦ Found zip: {zip_path}")
        
        # Step 3: Extract to proper location
        extract_dir = Path("/content") / project_name
        extract_dir.mkdir(exist_ok=True)
        
        print(f"ðŸ“‚ Extracting to: {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Step 4: Verify extraction
        src_path = extract_dir / "src" / "align_rerank"
        if not src_path.exists():
            # Try alternative: extracted directly to /content
            if Path("/content/src/align_rerank").exists():
                extract_dir = Path("/content")
                src_path = extract_dir / "src" / "align_rerank"
                print("âš ï¸  Found files in /content (flat extraction)")
            else:
                status["error"] = "Project structure not found after extraction"
                print(f"âŒ {status['error']}")
                return status
        
        # Step 5: Add to Python path
        src_parent = src_path.parent  # src/
        if str(src_parent) not in sys.path:
            sys.path.insert(0, str(src_parent))
        
        # Step 6: Create output directories
        runs_dir = extract_dir / "runs"
        results_dir = extract_dir / "results"
        runs_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        
        # Step 7: Verify key files
        key_files = [
            "score_factcc.py",
            "create_baseline.py",
            "verifiers/qags.py",
            "verifiers/factcc.py"
        ]
        missing = []
        for f in key_files:
            if not (src_path / f).exists():
                missing.append(f)
        
        # Step 8: Test import
        try:
            import align_rerank
            import_test = True
        except Exception as e:
            import_test = False
            print(f"âš ï¸  Import test failed: {e}")
        
        # Report status
        print("\n" + "="*50)
        print("ðŸ“Š Setup Summary")
        print("="*50)
        print(f"âœ… Project path: {extract_dir}")
        print(f"âœ… Source path: {src_path}")
        print(f"âœ… Python path: {src_parent} added")
        print(f"âœ… Output dirs: {runs_dir}, {results_dir}")
        
        if missing:
            print(f"âš ï¸  Missing files: {missing}")
        else:
            print("âœ… All key files present")
        
        if import_test:
            print("âœ… Module import successful")
        else:
            print("âš ï¸  Import test failed (check manually)")
        
        print("="*50 + "\n")
        
        status["success"] = True
        status["project_path"] = str(extract_dir)
        status["src_path"] = str(src_path)
        
        return status
        
    except Exception as e:
        status["error"] = str(e)
        print(f"âŒ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return status


def _find_zip_file(drive_base: str, project_name: str) -> Optional[str]:
    """Search for zip file in Google Drive."""
    import os
    
    # Common locations
    search_paths = [
        os.path.join(drive_base, f"{project_name}-updated.zip"),
        os.path.join(drive_base, f"{project_name}.zip"),
        os.path.join(drive_base, "W266-NLP", "266-NLP_Final_Project", f"{project_name}-updated.zip"),
    ]
    
    # Check common paths first
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    # Recursive search (limit depth to avoid long search)
    print("ðŸ” Searching for zip file...")
    for root, dirs, files in os.walk(drive_base):
        # Limit search depth
        depth = root[len(drive_base):].count(os.sep)
        if depth > 3:  # Max 3 levels deep
            dirs[:] = []  # Don't descend further
            continue
            
        for file in files:
            if file.endswith('.zip') and project_name.lower() in file.lower():
                return os.path.join(root, file)
    
    return None


if __name__ == "__main__":
    # For direct execution
    result = setup_colab()
    if not result["success"]:
        sys.exit(1)

