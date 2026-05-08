import os
import json
import numpy as np
import requests
from recbole.config import Config
from recbole.data.dataset import Dataset
from tqdm import tqdm


OLLAMA_GEN_MODEL = "qwen3.5:4b"
OLLAMA_EMBED_MODEL = "qwen3-embedding:8b"
OLLAMA_URL = "http://localhost:11434/api"
    
# The single prompt for the LLM
VIBE_PROMPT = """You are an expert consumer behavior analyst. What is the 'vibe' of this item? 
What kind of person would buy it? What are their demographics, lifestyle, and values? 
Keep it to 3 concise sentences. Item details: {text}"""

# The instruction prefix for the embedding model (Strategy 2)
TASK_PREFIX = "Represent this product for predicting collaborative purchase behavior and user lifestyle alignment: "

# Global cache for instant lookups
ITEM_TEXT_MAP = {}

def initialize_text_mapping(dataset_name="amazon-office"):
    """Bridges the gap between PyTorch IDs and Amazon ASINs."""
    global ITEM_TEXT_MAP
    print("\n--- 🗺️ INITIALIZING RECBOLE TEXT MAPPING ---")
    
    # 1. Ask RecBole how it mapped the IDs for this specific run
    config = Config(model='BPR', dataset=dataset_name)
    dataset = Dataset(config)
    id2token = dataset.field2id_token['item_id']
    
    # 2. Load your clean dictionary from setup_amazon.py
    try:
        with open(f"dataset/{dataset_name}/asin_to_text.json", 'r') as f:
            asin_to_text = json.load(f)
    except FileNotFoundError:
        raise Exception("Could not find asin_to_text.json. Did you run setup_amazon.py first?")
        
    # 3. Marry them together
    for internal_id in range(1, dataset.item_num):
        amazon_asin = id2token[internal_id]
        ITEM_TEXT_MAP[internal_id] = asin_to_text.get(amazon_asin, "No description available.")
        
    print(f"✅ Mapped {len(ITEM_TEXT_MAP)} items perfectly to PyTorch indices.")
    return dataset.item_num

def get_raw_item_text(item_id):
    """Instant O(1) lookup function for the embedding loop."""
    return ITEM_TEXT_MAP.get(item_id, "No description available.")

def generate_text(prompt):
    payload = {"model": OLLAMA_GEN_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(f"{OLLAMA_URL}/generate", json=payload).json()
    
    # Check if Ollama threw an error (e.g., model not found)
    if 'error' in response:
        raise RuntimeError(f"❌ Ollama Text Generation Error: {response['error']}")
        
    return response['response']

def generate_embedding(text):
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text, "stream": False}
    response = requests.post(f"{OLLAMA_URL}/embeddings", json=payload).json()
    
    # Check if Ollama threw an error
    if 'error' in response:
        raise RuntimeError(f"❌ Ollama Embedding Error: {response['error']}")
        
    return response['embedding']

def unload_model(model_name):
    print(f"\nFlushing {model_name} from VRAM...")
    requests.post(f"{OLLAMA_URL}/generate", json={"model": model_name, "keep_alive": 0})

def run_pipeline(dataset_name="amazon-office", total_items=1000):
    print(f"=== 🏗️ RUNNING 4-AXIS ABLATION PIPELINE FOR {dataset_name} ===")
    total_items = initialize_text_mapping(dataset_name) 
    
    save_dir = f"dataset/{dataset_name}"
    save_dir = f"dataset/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    text_cache_file = os.path.join(save_dir, "generated_vibe_cache.json")
    
    # ==========================================
    # PHASE 1: TEXT GENERATION (LLM ONLY)
    # ==========================================

    # 1. DEFINE CACHE PATH
    vibe_cache_path = f"dataset/{dataset_name}/vibe_cache.json"

    # 2. LOAD EXISTING PROGRESS (The magic resume feature)
    if os.path.exists(vibe_cache_path):
        with open(vibe_cache_path, 'r') as f:
            vibe_cache = json.load(f)
        print(f"🔄 Resuming from {len(vibe_cache)} previously saved items!")
    else:
        vibe_cache = {}

    # 3. THE INCREMENTAL LOOP
    print("--- PHASE 1: GENERATING LLM VIBE DESCRIPTIONS ---")
    for i in tqdm(range(1, total_items)):
        str_id = str(i)
        
        # SKIP if we already generated it in a previous run!
        if str_id in vibe_cache:
            continue
            
        raw_text = get_raw_item_text(i)
        prompt = f"Analyze this product: {raw_text}. Describe the consumer psychology and vibe..."
        
        # Generate and save to dict
        vibe_cache[str_id] = generate_text(prompt)
        
        # EVERY 10 ITEMS: Save securely to disk so we never lose data
        if i % 10 == 0:
            with open(vibe_cache_path, 'w') as f:
                json.dump(vibe_cache, f)

    # Final save to catch the last few items
    with open(vibe_cache_path, 'w') as f:
        json.dump(vibe_cache, f)
        
    unload_model(OLLAMA_GEN_MODEL)

    # ==========================================
    # PHASE 2: MATH EMBEDDING (4 STRATEGIES)
    # ==========================================
    print("\n--- PHASE 2: GENERATING VECTORS FOR ALL 4 STRATEGIES ---")
    strategies = ["raw", "prefixed", "vibe_only", "combination"]
    
    # Initialize 4 distinct matrices
    embed_matrices = {s: np.zeros((total_items, 1024)) for s in strategies} # Change 1024 if bge-m3 differs!
    
    for item_id in tqdm(range(1, total_items), desc="Embedding all variations"):
        str_id = str(item_id)
        raw_text = get_raw_item_text(item_id)
        generated_vibe = vibe_cache.get(str_id, "")
        
        # 1. Raw Metadata
        embed_matrices["raw"][item_id] = generate_embedding(raw_text)
        
        # 2. Instruction-Prefixed
        embed_matrices["prefixed"][item_id] = generate_embedding(TASK_PREFIX + raw_text)
        
        # 3. LLM Vibe Only
        embed_matrices["vibe_only"][item_id] = generate_embedding(generated_vibe)
        
        # 4. Combination
        combo_text = f"Product: {raw_text}\nConsumer Profile: {generated_vibe}"
        embed_matrices["combination"][item_id] = generate_embedding(combo_text)
            
    # Save all 4 matrices
    print("\n--- SAVING CACHES TO DISK ---")
    for strategy in strategies:
        filename = os.path.join(save_dir, f"{dataset_name}_{strategy}_embeddings.npy")
        np.save(filename, embed_matrices[strategy])
        print(f"✅ Saved: {filename}")
        
    unload_model(OLLAMA_EMBED_MODEL)
    print("\n🎉 4-AXIS PIPELINE COMPLETE!")

if __name__ == "__main__":
    run_pipeline("amazon-office", total_items=500)