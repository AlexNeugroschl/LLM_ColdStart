import os
import json
import numpy as np
import requests
from recbole.config import Config
from recbole.data.dataset import Dataset
from tqdm import tqdm

# ==========================================
# 1. HARDWARE & SERVER CONFIGURATION
# ==========================================
# Blindfold PyTorch to prevent VRAM thrashing (Keep RecBole strictly on CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONUTF8"] = "1"

OLLAMA_GEN_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api"

# Global cache for instant memory lookups
ITEM_TEXT_MAP = {}

# ==========================================
# 2. DATASET-SPECIFIC LLM PROMPTS
# ==========================================
DATASET_PROMPTS = {
    'amazon-office': {
        'system': "You are an expert consumer psychology analyst.",
        'task_prefix': "Represent this physical office product for a semantic collaborative filtering system: ",
        'vibe_prompt': """Write a 3-sentence psychological profile of the buyer of this specific product. 
        
RULES:
1. Jump straight into the analysis. DO NOT start with "The vibe of this item is..." or "This item targets...".
2. ZERO generic demographics. You are strictly forbidden from using phrases like "aged 25-45", "moderate income", or "professionals".
3. Anchor the buyer's mindset entirely to the product's unique physical traits and specific use-case.

Product Details:
{text}

Analysis:"""
    },
    
    'ml-1m': {
        'system': "You are an expert film critic and audience psychoanalyst.",
        'task_prefix': "Represent this movie for a semantic collaborative filtering system: ",
        'vibe_prompt': """Write a 3-sentence psychological profile of the viewer who would love this specific movie. 
        
RULES:
1. Jump straight into the analysis. DO NOT start with "The vibe of this movie is..." or "This movie targets...".
2. ZERO generic demographics. You are strictly forbidden from using phrases like "aged 25-45", "families", or "general audiences".
3. Anchor the viewer's mindset entirely to the movie's specific narrative traits, genres, and the emotional payoff they are seeking.

Movie Details:
{text}

Analysis:"""
    },
    'ml-100k': {
        'system': "You are an expert film critic and audience psychoanalyst.",
        'task_prefix': "Represent this movie for a semantic collaborative filtering system: ",
        'vibe_prompt': """Write a 3-sentence psychological profile of the viewer who would love this specific movie. 
        
RULES:
1. Jump straight into the analysis. DO NOT start with "The vibe of this movie is..." or "This movie targets...".
2. ZERO generic demographics. You are strictly forbidden from using phrases like "aged 25-45", "families", or "general audiences".
3. Anchor the viewer's mindset entirely to the movie's specific narrative traits, genres, and the emotional payoff they are seeking.

Movie Details:
{text}

Analysis:"""
    },
    
    'steam': {
        'system': "You are an expert gaming psychologist and player-behavior analyst.",
        'task_prefix': "Represent this video game for a semantic collaborative filtering system: ",
        'vibe_prompt': """Write a 3-sentence psychological profile of the core player of this specific video game. 
        
RULES:
1. Jump straight into the analysis. DO NOT start with "The vibe of this game is..." or "This game targets...".
2. ZERO generic demographics. You are strictly forbidden from using phrases like "aged 15-35", "teens", or "casual gamers".
3. Anchor the player's mindset entirely to the game's unique mechanics, tags, and the specific gameplay challenge or experience they crave.

Game Details:
{text}

Analysis:"""
    },
    'amazon-digital-music': {
        'system': "You are an expert music psychologist and audiophile.",
        'task_prefix': "Represent this digital album for a semantic collaborative filtering system: ",
        'vibe_prompt': """Write a 3-sentence psychological profile of the listener who would buy this specific album. 
        
RULES:
1. Jump straight into the analysis. DO NOT start with "The vibe of this album is..." or "This music targets...".
2. ZERO generic demographics. You are strictly forbidden from using phrases like "aged 18-35", "teens", or "general audiences".
3. Anchor the listener's mindset entirely to the album's specific sonic traits, genres, and the emotional or aesthetic resonance they are seeking.

Album Details:
{text}

Analysis:"""
    },
}

# ==========================================
# 3. CORE UTILITIES
# ==========================================
def initialize_text_mapping(dataset_name):
    """Universal loader that reads the JSON created by extract_text.py"""
    global ITEM_TEXT_MAP
    ITEM_TEXT_MAP.clear()
    print(f"\n--- 🗺️ INITIALIZING TEXT MAPPING FOR {dataset_name} ---")
    
    config = Config(
        model='LightGCN', 
        dataset=dataset_name,
        config_file_list=['configs/base_recbole.yaml'],
        config_dict={'encoding': 'utf-8'}
    )
    
    dataset = Dataset(config)
    id2token = dataset.field2id_token['item_id']
    json_path = f"dataset/{dataset_name}/item_to_text.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            item_to_text = json.load(f)
    except FileNotFoundError:
        raise Exception(f"❌ Could not find {json_path}. Did you run extract_text.py first?")
        
    for internal_id in range(1, dataset.item_num):
        original_id = id2token[internal_id]
        ITEM_TEXT_MAP[internal_id] = item_to_text.get(original_id, "No description available.")
        
    print(f"✅ Mapped {len(ITEM_TEXT_MAP)} items perfectly to PyTorch indices.")
    return dataset.item_num

def get_raw_item_text(item_id):
    """Instant O(1) lookup function for the embedding loop."""
    return ITEM_TEXT_MAP.get(item_id, "No description available.")

def generate_text(item_text, system_prompt, prompt_template):
    """Dynamically generates the psychological vibe based on the dataset prompt."""
    combined_prompt = prompt_template.format(text=item_text)
    
    payload = {
        "model": OLLAMA_GEN_MODEL, 
        "messages": [
            {"role": "system", "content": system_prompt},
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

def generate_batch_embeddings(texts):
    """Passes an array of strings to Ollama to embed them all simultaneously on the GPU."""
    payload = {
        "model": OLLAMA_EMBED_MODEL, 
        "input": texts,       # ⬅️ Passing an array of 4 strings!
        "keep_alive": "10m",
        "options": {
            "num_ctx": 8192  
        }
    }
    # ⬅️ Using the new /embed batch endpoint
    response = requests.post(f"{OLLAMA_URL}/embed", json=payload).json()
    
    if 'error' in response:
        raise RuntimeError(f"❌ Ollama Embedding Error: {response['error']}")
        
    return response['embeddings'] # Returns a list of 4 vectors

def unload_model(model_name):
    """Frees up GPU VRAM when switching models."""
    print(f"\nFlushing {model_name} from VRAM...")
    requests.post(f"{OLLAMA_URL}/generate", json={"model": model_name, "keep_alive": 0})

# ==========================================
# 4. THE MAIN PIPELINE EXECUTION
# ==========================================
def run_pipeline(dataset_name="amazon-office"):
    print(f"=== 🏗️ RUNNING 4-AXIS ABLATION PIPELINE FOR {dataset_name} ===")
    
    # Check if dataset's text cache exists before proceeding
    text_cache_path = f"dataset/{dataset_name}/item_to_text.json"
    if not os.path.exists(text_cache_path):
        print(f"⏭️  SKIPPING {dataset_name}: {text_cache_path} not found.")
        print(f"    (Run setup + extract_text.py first for this dataset.)\n")
        return
    
    try:
        total_items = initialize_text_mapping(dataset_name)
    except FileNotFoundError as e:
        print(f"⏭️  SKIPPING {dataset_name}: {e}\n")
        return
    
    # Grab the specific prompts for this domain
    prompts = DATASET_PROMPTS.get(dataset_name, DATASET_PROMPTS['amazon-office'])
    
    save_dir = f"dataset/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # ------------------------------------------
    # PHASE 1: TEXT GENERATION (LLM ONLY)
    # ------------------------------------------
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
            
        safe_text = str(get_raw_item_text(i))[:1500] 
        vibe_cache[str_id] = generate_text(safe_text, prompts['system'], prompts['vibe_prompt'])
        
        if i % 10 == 0:
            with open(vibe_cache_path, 'w') as f:
                json.dump(vibe_cache, f)

    with open(vibe_cache_path, 'w') as f:
        json.dump(vibe_cache, f)
        
    unload_model(OLLAMA_GEN_MODEL)

    # ------------------------------------------
    # PHASE 2: MATH EMBEDDING (4 STRATEGIES)
    # ------------------------------------------
    print("\n--- PHASE 2: GENERATING VECTORS FOR ALL 4 STRATEGIES ---")
    strategies = ["raw", "prefixed", "vibe_only", "combination"]
    
    embed_matrices = {}
    matrix_paths = {}
    prefix = prompts['task_prefix']
    
    # Load existing caches to RAM
    for s in strategies:
        path = f"dataset/{dataset_name}/{dataset_name}_{s}_embeddings.npy"
        matrix_paths[s] = path
        
        if os.path.exists(path):
            print(f"🔄 Resuming existing matrix: {path}")
            embed_matrices[s] = np.load(path)
        else:
            embed_matrices[s] = np.zeros((total_items, 768))
            
    # Try/Finally block to intercept crashes and save progress safely
    try:
        BATCH_SIZE = 16  # Process 16 items (64 embeddings) at once!
        
        # Create a list of all item IDs we need to process
        all_items = list(range(1, total_items))
        
        # Iterate through the items in chunks
        for i in tqdm(range(0, len(all_items), BATCH_SIZE), desc="Batch Embedding"):
            batch_ids = all_items[i : i + BATCH_SIZE]
            
            texts_to_embed = []
            valid_ids = []
            
            # 1. Build the massive array for this chunk
            for item_id in batch_ids:
                # Skip if already embedded
                if np.any(embed_matrices["raw"][item_id]):
                    continue
                    
                str_id = str(item_id)
                raw_text = str(get_raw_item_text(item_id))[:4000] 
                generated_vibe = vibe_cache.get(str_id, "")
                combo_text = f"Product: {raw_text}\nConsumer Profile: {generated_vibe}"
                
                # Append the 4 strings to our master batch list
                texts_to_embed.extend([
                    raw_text,
                    prefix + raw_text,
                    generated_vibe,
                    combo_text
                ])
                valid_ids.append(item_id)
                
            # 2. Skip API call if the whole chunk was already cached
            if not texts_to_embed:
                continue
                
            # 3. Hit the Batch API (One massive GPU forward pass)
            batch_vectors = generate_batch_embeddings(texts_to_embed)
            
            # 4. Unpack the linear results back into the matrices
            idx = 0
            for item_id in valid_ids:
                embed_matrices["raw"][item_id] = batch_vectors[idx]
                embed_matrices["prefixed"][item_id] = batch_vectors[idx + 1]
                embed_matrices["vibe_only"][item_id] = batch_vectors[idx + 2]
                embed_matrices["combination"][item_id] = batch_vectors[idx + 3]
                idx += 4

    finally:
        print("\n💾 Saving Phase 2 matrices to disk...")
        for s in strategies:
            np.save(matrix_paths[s], embed_matrices[s])
        print("✅ Safe shutdown complete.")
        
    unload_model(OLLAMA_EMBED_MODEL)
    print(f"\n🎉 4-AXIS PIPELINE COMPLETE FOR {dataset_name.upper()}!")

# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    datasets_to_run = [
        "amazon-office",
        "amazon-digital-music",
        "ml-1m",
        "ml-100k",
        # "steam"
    ]
    
    for dataset in datasets_to_run:
        print(f"\n\n{'='*50}")
        print(f"🚀 STARTING PIPELINE FOR: {dataset}")
        print(f"{'='*50}")
        run_pipeline(dataset)