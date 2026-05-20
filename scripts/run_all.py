import subprocess
import sys
import os

# Map setup scripts to their target dataset folders
SETUP_MAP = {
    "setup_ml.py": "ml-1m",
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
    
    # If the script throws an error, kill the pipeline immediately
    if result.returncode != 0:
        print(f"\n❌ FATAL ERROR: {script_name} crashed! Halting the master pipeline.")
        sys.exit(1)
        
    print(f"\n✅ SUCCESS: {script_name} finished flawlessly.\n")

if __name__ == "__main__":
    # The exact execution order
    pipeline = [
        "setup_amazon.py",
        "setup_ml.py",
        "setup_steam.py",
        "split_data.py",
        "extract_text.py",
        "build_semantic_caches.py" 
    ]
    
    print("=== 🌟 STARTING OVERNIGHT MASTER PIPELINE 🌟 ===")
    
    for script in pipeline:
        run_script(script)
        
    print("\n🎉 OVERNIGHT RUN COMPLETE! All datasets are split, extracted, and embedded.")