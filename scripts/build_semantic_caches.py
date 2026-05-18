import os
import json
import numpy as np
import requests
from recbole.config import Config
from recbole.data.dataset import Dataset
from tqdm import tqdm

# --- 1. BLINDFOLD PYTORCH (Keep it strictly on CPU) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 2. CONFIGURATION ---
OLLAMA_GEN_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api"
    
# --- 3. THE SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a highly observant consumer psychology analyst. 
Analyze the unique aesthetic, emotional 'vibe', and specific use-case of this exact product. 
Describe the lifestyle and specific mindset it appeals to. 
RULES: 
1. Explicitly name the type of item.
2. DO NOT use generic demographic brackets (e.g., avoid "aged 25-45" or "moderate income").
3. Keep it to exactly 3 concise sentences."""

# The instruction prefix for the embedding model (Strategy 2)
TASK_PREFIX = "Represent this product for predicting collaborative purchase behavior and user lifestyle alignment: "

# Global cache for instant lookups
ITEM_TEXT_MAP = {}

def initialize_text_mapping(dataset_name="amazon-office"):
    """Bridges the gap between PyTorch IDs and Amazon ASINs."""
    global ITEM_TEXT_MAP
    print("\n--- 🗺️ INITIALIZING RECBOLE TEXT MAPPING ---")
    
    config = Config(model='BPR', dataset=dataset_name)
    dataset = Dataset(config)
    id2token = dataset.field2id_token['item_id']
    
    try:
        with open(f"dataset/{dataset_name}/asin_to_text.json", 'r') as f:
            asin_to_text = json.load(f)
    except FileNotFoundError:
        raise Exception("Could not find asin_to_text.json. Did you run setup_amazon.py first?")
        
    for internal_id in range(1, dataset.item_num):
        amazon_asin = id2token[internal_id]
        ITEM_TEXT_MAP[internal_id] = asin_to_text.get(amazon_asin, "No description available.")
        
    print(f"✅ Mapped {len(ITEM_TEXT_MAP)} items perfectly to PyTorch indices.")
    return dataset.item_num

def get_raw_item_text(item_id):
    """Instant O(1) lookup function for the embedding loop."""
    return ITEM_TEXT_MAP.get(item_id, "No description available.")

def generate_text(item_text):
    # The ultimate anti-horoscope, anti-robot prompt
    combined_prompt = f"""You are a consumer psychology analyst. Write a 3-sentence psychological profile of the buyer of this specific product. 
        
RULES:
1. Jump straight into the analysis. DO NOT start with "The vibe of this item is..." or "This item targets...".
2. ZERO generic demographics. You are strictly forbidden from using phrases like "aged 25-45", "moderate income", or "professionals".
3. Anchor the buyer's mindset entirely to the product's unique physical traits and specific use-case.

Product Details:
{item_text}

Analysis:"""

    payload = {
        "model": OLLAMA_GEN_MODEL, 
        "messages": [
            {"role": "user", "content": combined_prompt}
        ], 
        "stream": False,
        "options": {
            "num_ctx": 2048,      
            "temperature": 0.0    
        }
    }
    response = requests.post(f"{OLLAMA_URL}/chat", json=payload).json()
    
    if 'error' in response:
        raise RuntimeError(f"❌ Ollama Chat Error: {response['error']}")
        
    return response['message']['content']

def generate_embedding(text):
    payload = {
        "model": OLLAMA_EMBED_MODEL, 
        "prompt": text, 
        "stream": False,
        "options": {
            "num_ctx": 8192  # ⬅️ Force Ollama to use Nomic's maximum context window
        }
    }
    response = requests.post(f"{OLLAMA_URL}/embeddings", json=payload).json()
    
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
    os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # PHASE 1: TEXT GENERATION (LLM ONLY)
    # ==========================================
    vibe_cache_path = os.path.join(save_dir, "vibe_cache.json")

    if os.path.exists(vibe_cache_path):
        with open(vibe_cache_path, 'r') as f:
            vibe_cache = json.load(f)
        print(f"🔄 Resuming from {len(vibe_cache)} previously saved items!")
    else:
        vibe_cache = {}

    print("--- PHASE 1: GENERATING LLM VIBE DESCRIPTIONS ---")
    for i in tqdm(range(1, total_items)):
        str_id = str(i)
        
        if str_id in vibe_cache:
            continue
            
        # 🛑 The Golden Balance: ~250 words fits easily into 2048 tokens.
        # No rsplit needed, the tokenizer handles chopped words perfectly.
        safe_text = str(get_raw_item_text(i))[:1500] 
        
        vibe_cache[str_id] = generate_text(safe_text)
        
        if i % 10 == 0:
            with open(vibe_cache_path, 'w') as f:
                json.dump(vibe_cache, f)

    with open(vibe_cache_path, 'w') as f:
        json.dump(vibe_cache, f)
        
    unload_model(OLLAMA_GEN_MODEL)

    # ==========================================
    # PHASE 2: MATH EMBEDDING (4 STRATEGIES)
    # ==========================================
    print("\n--- PHASE 2: GENERATING VECTORS FOR ALL 4 STRATEGIES ---")
    strategies = ["raw", "prefixed", "vibe_only", "combination"]
    
    embed_matrices = {s: np.zeros((total_items, 768)) for s in strategies} 
    
    for item_id in tqdm(range(1, total_items), desc="Embedding all variations"):
        str_id = str(item_id)
        
        raw_text = str(get_raw_item_text(item_id))[:4000] 
        
        generated_vibe = vibe_cache.get(str_id, "")
        
        embed_matrices["raw"][item_id] = generate_embedding(raw_text)
        embed_matrices["prefixed"][item_id] = generate_embedding(TASK_PREFIX + raw_text)
        embed_matrices["vibe_only"][item_id] = generate_embedding(generated_vibe)
        
        combo_text = f"Product: {raw_text}\nConsumer Profile: {generated_vibe}"
        embed_matrices["combination"][item_id] = generate_embedding(combo_text)
            
    print("\n--- SAVING CACHES TO DISK ---")
    for strategy in strategies:
        filename = os.path.join(save_dir, f"{dataset_name}_{strategy}_embeddings.npy")
        np.save(filename, embed_matrices[strategy])
        print(f"✅ Saved: {filename}")
        
    unload_model(OLLAMA_EMBED_MODEL)
    print("\n🎉 4-AXIS PIPELINE COMPLETE!")

if __name__ == "__main__":
    run_pipeline("amazon-office")