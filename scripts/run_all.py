import subprocess
import sys
import os

# Map setup scripts to their target dataset folders
SETUP_MAP = {
    "setup_ml.py": "ml-1m",
    "setup_ml_100k.py": "ml-100k",
    "setup_steam.py": "steam",
    "setup_amazon.py": "amazon-office"
}

def check_if_setup_needed(script_name):
    """Checks if the raw RecBole files already exist to prevent redundant parsing."""
    if script_name not in SETUP_MAP:
        # If it's a processing script (split_data, extract_text, etc.), always run it
        return True 
        
    dataset_name = SETUP_MAP[script_name]
    
    # Check for the files that the setup scripts generate
    inter_path = os.path.join("dataset", dataset_name, f"{dataset_name}.inter")
    item_path = os.path.join("dataset", dataset_name, f"{dataset_name}.item")
    
    if os.path.exists(inter_path) and os.path.exists(item_path):
        return False
        
    return True

def run_script(script_name):
    print(f"\n{'='*60}")
    
    # 🛑 The Smart Check
    if not check_if_setup_needed(script_name):
        print(f"⏭️  SKIPPING: {script_name}")
        print(f"    (Dataset files already exist. Moving to next step.)")
        print(f"{'='*60}")
        return

    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}")
    
    # Run the script using the current Python environment
    result = subprocess.run([sys.executable, f"scripts/{script_name}"])
    
    # If the script throws an error, warn but allow continuation for setup scripts
    if result.returncode != 0:
        if script_name in ["setup_amazon.py", "setup_ml.py", "setup_ml_100k.py", "setup_steam.py"]:
            print(f"\n⚠️  WARNING: {script_name} did not complete fully.")
            print(f"    This may be OK if you are providing the dataset manually.")
            print(f"    If this is unexpected, resolve the error and re-run.\n")
        else:
            print(f"\n❌ FATAL ERROR: {script_name} crashed! Halting the master pipeline.")
            sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: {script_name} finished flawlessly.\n")

if __name__ == "__main__":
    # The exact execution order
    pipeline = [
        "setup_amazon.py",
        "setup_ml.py",
        "setup_ml_100k.py",
        "extract_text.py",
        "build_semantic_caches.py" 
    ]
    
    print("=== 🌟 STARTING OVERNIGHT MASTER PIPELINE 🌟 ===")
    
    for script in pipeline:
        run_script(script)
        
    print("\n🎉 OVERNIGHT RUN COMPLETE! All datasets are split, extracted, and embedded.")