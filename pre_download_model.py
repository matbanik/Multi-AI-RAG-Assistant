import os
import sys
from huggingface_hub import snapshot_download

# This script downloads the specified Hugging Face model to a local directory.
# It's designed to be run by the build script before packaging the application.

def download_model():
    """Downloads the model to a predictable local directory."""
    # The full model name includes the 'sentence-transformers' organization.
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # The local path should still be simple for PyInstaller to handle easily.
    # We'll strip the organization part for the folder name.
    simple_name = model_name.split('/')[-1]
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    model_save_path = os.path.join(script_dir, 'local_model', simple_name)

    print(f"--- Preparing to download embedding model: {model_name} ---")

    if not os.path.exists(model_save_path):
        print(f"Model not found locally. Downloading to: {model_save_path}")
        try:
            # CORRECTED: Using the 'sha' value from the Hugging Face API response
            # provided by the user. This is the correct latest commit hash.
            snapshot_download(
                repo_id=model_name,
                local_dir=model_save_path,
                local_dir_use_symlinks=False,
                revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf" 
            )
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to download model. {e}")
            sys.exit(1) # Exit with an error code
    else:
        print(f"Model already exists at: {model_save_path}")
    
    print("--- Pre-download complete ---")


if __name__ == "__main__":
    download_model()

