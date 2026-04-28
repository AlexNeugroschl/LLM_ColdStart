import os
import copy
import numpy as np
import concurrent.futures
import torch
import faiss
import ollama
from tqdm import tqdm # pip install tqdm (Highly recommended for progress tracking)

def apply_knn(payload, threshold=5, k_neighbors=5, embed_model='mxbai-embed-large'):
    print(f"\n--- Applying FAISS KNN Semantic Imputation (Threshold < {threshold}) ---")
    state = copy.copy(payload)
    model = state['model']
    dataset = state['dataset']
    ds_name = state['config']['dataset']
    
    # ---------------------------------------------------------
    # 1. IDENTIFY WARM VS. COLD ITEMS
    # ---------------------------------------------------------
    # RecBole reserves item_id 0 for padding. We ignore it.
    item_num = dataset.item_num
    
    # Count occurrences in the training set to prevent data leakage
    train_inter = state['train_data'].dataset.inter_feat
    item_counts = np.bincount(train_inter['item_id'].numpy(), minlength=item_num)
    
    # Identify IDs based on your tunable threshold (ignoring ID 0)
    warm_item_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_item_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]
    
    print(f"Dataset stats: {len(warm_item_ids)} Warm items | {len(cold_item_ids)} Cold items.")
    
    if len(cold_item_ids) == 0:
        print("No cold items found based on threshold. Skipping imputation.")
        return state

    # ---------------------------------------------------------
    # 2. GENERATE OR LOAD SEMANTIC EMBEDDINGS
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. GENERATE OR LOAD SEMANTIC EMBEDDINGS (PARALLELIZED)
    # ---------------------------------------------------------
    
    cache_file = f"dataset/{ds_name}/{ds_name}_{embed_model}_embeddings.npy"
    
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        all_embeddings = np.load(cache_file)
    else:
        print(f"Cache not found. Generating embeddings via Ollama ({embed_model})...")
        
        sample_response = ollama.embeddings(model=embed_model, prompt="test")
        embed_dim = len(sample_response['embedding'])
        all_embeddings = np.zeros((item_num, embed_dim), dtype=np.float32)
        
        def decode_feature(dataset, field, item_id):
            if field not in dataset.item_feat: return "Unknown"
            tensor_val = dataset.item_feat[field][item_id]
            if tensor_val.dim() == 0:  
                tid = tensor_val.item()
                return str(dataset.id2token(field, tid)) if tid != 0 else "Unknown"
            else:  
                tids = [t for t in tensor_val.tolist() if t != 0]
                if not tids: return "Unknown"
                return " ".join([str(dataset.id2token(field, t)) for t in tids])

        # Step A: Pre-build all the text strings
        print("Pre-processing text descriptions...")
        item_texts = {}
        for item_id in range(1, item_num):
            if ds_name == 'ml-100k':
                title = decode_feature(dataset, 'movie_title', item_id)
                genres = decode_feature(dataset, 'class', item_id)
                text = f"Movie Title: {title}. Genres: {genres}."
            elif ds_name == 'amazon-books':
                title = decode_feature(dataset, 'title', item_id)
                categories = decode_feature(dataset, 'categories', item_id)
                text = f"Book Title: {title}. Categories: {categories}."
            else: # Yelp
                name = decode_feature(dataset, 'item_name', item_id)
                categories = decode_feature(dataset, 'categories', item_id)
                city = decode_feature(dataset, 'city', item_id)
                text = f"Business: {name}. Categories: {categories}. City: {city}."
            
            item_texts[item_id] = text

        # Step B: Worker function for the thread pool
        def fetch_embedding(item_id, text):
            response = ollama.embeddings(model=embed_model, prompt=text)
            return item_id, response['embedding']

        # Step C: Blast Ollama with parallel requests (Adjust max_workers if needed)
        print("Firing concurrent API requests to local Ollama GPU...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit all tasks
            futures = [executor.submit(fetch_embedding, i, t) for i, t in item_texts.items()]
            
            # Process them as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Ollama Parallel"):
                item_id, emb = future.result()
                all_embeddings[item_id] = emb
            
        print(f"Saving embeddings to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, all_embeddings)

    # ---------------------------------------------------------
    # 3. BUILD THE FAISS INDEX (Cosine Similarity)
    # ---------------------------------------------------------
    print("Building FAISS index with warm items...")
    warm_embeddings = all_embeddings[warm_item_ids].copy()
    cold_embeddings = all_embeddings[cold_item_ids].copy()
    
    # Normalize vectors for Cosine Similarity (IndexFlatIP is Inner Product)
    faiss.normalize_L2(warm_embeddings)
    faiss.normalize_L2(cold_embeddings)
    
    embed_dim = warm_embeddings.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    index.add(warm_embeddings)
    
    # ---------------------------------------------------------
    # 4. QUERY FAISS AND IMPUTE PYTORCH WEIGHTS
    # ---------------------------------------------------------
    print(f"Querying top {k_neighbors} neighbors for cold items...")
    distances, indices = index.search(cold_embeddings, k=k_neighbors)
    
    print("Overwriting PyTorch embedding weights...")
    with torch.no_grad():
        for i, cold_id in enumerate(tqdm(cold_item_ids, desc="Imputing")):
            # indices[i] returns the position in the warm_item_ids array
            # We map that back to the actual PyTorch item_ids
            neighbor_actual_ids = warm_item_ids[indices[i]]
            
            # Fetch the learned collaborative filtering weights of the warm neighbors
            neighbor_weights = model.item_embedding.weight.data[neighbor_actual_ids]
            
            # Average them and overwrite the cold item's weight
            imputed_weight = torch.mean(neighbor_weights, dim=0)
            model.item_embedding.weight.data[cold_id] = imputed_weight

    print("KNN Semantic Imputation complete!")
    state['model'] = model
    return state