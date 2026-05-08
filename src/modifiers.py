import os
import copy
import numpy as np
import concurrent.futures
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import ollama
from tqdm import tqdm


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
            elif ds_name in ['amazon-office', 'amazon-digital-music']:
                title = decode_feature(dataset, 'title', item_id)
                categories = decode_feature(dataset, 'categories', item_id)
                brand = decode_feature(dataset, 'brand', item_id)
                description = decode_feature(dataset, 'description', item_id)
                
                # The optimized semantic payload (Prompt Engineered)
                text = (
                    f"Represent this Amazon product for recommendation based on its core category and utility:\n"
                    f"Brand: {brand}\n"
                    f"Title: {title}\n"
                    f"Category: {categories}\n"
                    f"Key Details: {description[:300]}" # Truncate marketing fluff
                )
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

def apply_tfidf_knn(payload, threshold=5, k_neighbors=5):
    print(f"\n--- Applying TF-IDF Baseline Imputation (Threshold < {threshold}) ---")
    state = copy.copy(payload)
    model = state['model']
    dataset = state['dataset']
    ds_name = state['config']['dataset']
    
    item_num = dataset.item_num
    train_inter = state['train_data'].dataset.inter_feat
    item_counts = np.bincount(train_inter['item_id'].numpy(), minlength=item_num)
    
    warm_item_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_item_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]
    
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

    # 1. Build Text Corpus
    print("Pre-processing text for TF-IDF...")
    item_texts = []
    for item_id in range(item_num):
        if item_id == 0:
            item_texts.append("")
            continue
            
        title = decode_feature(dataset, 'title', item_id)
        categories = decode_feature(dataset, 'categories', item_id)
        brand = decode_feature(dataset, 'brand', item_id)
        description = decode_feature(dataset, 'description', item_id)
        
        # Exact same text payload as the LLM for a fair fight!
        text = f"{brand} {title} {categories} {description[:300]}"
        item_texts.append(text)

    # 2. Generate TF-IDF Vectors
    print("Vectorizing corpus using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    all_vectors = vectorizer.fit_transform(item_texts)

    warm_vectors = all_vectors[warm_item_ids]
    cold_vectors = all_vectors[cold_item_ids]

    # 3. Compute Cosine Similarity & Impute
    print(f"Calculating Top-{k_neighbors} neighbors and overwriting weights...")
    similarity_matrix = cosine_similarity(cold_vectors, warm_vectors)
    
    with torch.no_grad():
        for i, cold_id in enumerate(tqdm(cold_item_ids, desc="Imputing")):
            # Get indices of top K most similar warm items
            top_k_indices = np.argsort(similarity_matrix[i])[-k_neighbors:]
            neighbor_actual_ids = warm_item_ids[top_k_indices]
            
            neighbor_weights = model.item_embedding.weight.data[neighbor_actual_ids]
            model.item_embedding.weight.data[cold_id] = torch.mean(neighbor_weights, dim=0)

    print("TF-IDF Imputation complete!")
    state['model'] = model
    return state



# =========================================================
# THE PROJECTION NETWORK (Now with LayerNorm)
# =========================================================
class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingMapper, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),         # <--- Forces variance, fights mode collapse
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),         # <--- Fights mode collapse
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def apply_contrastive_mapper(payload, threshold=5, embed_model='mxbai-embed-large', epochs=100):
    print(f"\n--- Training InfoNCE Projection Network (Epochs: {epochs}) ---")
    state = copy.copy(payload)
    model = state['model']
    dataset = state['dataset']
    ds_name = state['config']['dataset']
    
    item_num = dataset.item_num
    train_inter = state['train_data'].dataset.inter_feat
    item_counts = np.bincount(train_inter['item_id'].numpy(), minlength=item_num)
    
    warm_item_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_item_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]

    cache_file = f"dataset/{ds_name}/{ds_name}_{embed_model}_embeddings.npy"
    all_embeddings = np.load(cache_file)
    
    device = model.device
    X_warm = torch.tensor(all_embeddings[warm_item_ids], dtype=torch.float32).to(device)
    X_cold = torch.tensor(all_embeddings[cold_item_ids], dtype=torch.float32).to(device)
    Y_warm_true = model.item_embedding.weight.data[warm_item_ids].clone().detach().to(device)
    
    # 1. Normalize Inputs (Fix 4)
    X_warm = torch.nn.functional.normalize(X_warm, p=2, dim=1)
    X_cold = torch.nn.functional.normalize(X_cold, p=2, dim=1)

    mapper = EmbeddingMapper(X_warm.shape[1], Y_warm_true.shape[1]).to(device)
    
    # We use CrossEntropy for InfoNCE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mapper.parameters(), lr=0.001)
    
    # Batch size needs to be large for InfoNCE so there are many negatives
    batch_size = min(1024, len(warm_item_ids)) 
    train_dataset = data.TensorDataset(X_warm, Y_warm_true)
    dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Learning mapping via InfoNCE (CLIP) Loss...")
    mapper.train()
    temperature = 0.07 # Standard temperature for InfoNCE
    
    for epoch in tqdm(range(epochs), desc="Training Mapper"):
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            
            # 1. Predict
            anchor = mapper(batch_X)
            
            # 2. Normalize both to unit sphere to isolate Angles
            pred_norm = nn.functional.normalize(anchor, p=2, dim=1)
            actual_norm = nn.functional.normalize(batch_Y, p=2, dim=1)
            
            # 3. InfoNCE Similarity Matrix (Every item vs Every item)
            sim_matrix = torch.matmul(pred_norm, actual_norm.T) / temperature
            
            # The correct target is the diagonal (Item i should match Item i)
            labels = torch.arange(len(batch_X)).to(device)
            
            loss = criterion(sim_matrix, labels)
            loss.backward()
            optimizer.step()
            
    print("Injecting Cold Items via InfoNCE Projection...")
    mapper.eval()
    with torch.no_grad():
        Y_cold_imputed = mapper(X_cold)
        
        # Safe Median Magnitude Restoration (Preserves Popularity Physics)
        target_magnitude = torch.norm(Y_warm_true, p=2, dim=1).median()
        Y_cold_imputed = torch.nn.functional.normalize(Y_cold_imputed, p=2, dim=1) * target_magnitude
        
        model.item_embedding.weight.data[cold_item_ids] = Y_cold_imputed
        
    state['model'] = model
    return state