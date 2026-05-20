import os
import copy
import json
import numpy as np
import concurrent.futures
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import ollama
from tqdm import tqdm
import scipy.sparse as sp



def apply_tfidf_knn(payload, threshold=5, k_neighbors=1, strategy_name="raw"):
    print(f"\n--- Applying TF-IDF Baseline Imputation (Threshold < {threshold}, K={k_neighbors}, Strategy={strategy_name.upper()}) ---")
    state = copy.copy(payload)
    model = state['model']
    dataset = state['dataset']
    ds_name = state['config']['dataset']
    
    item_num = dataset.item_num
    train_inter = state['train_data'].dataset.inter_feat
    item_counts = np.bincount(train_inter['item_id'].cpu().numpy(), minlength=item_num)
    
    warm_item_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_item_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]
    
    # 1. LOAD THE TEXT DICTIONARIES BASED ON STRATEGY
    raw_json_path = f"dataset/{ds_name}/item_to_text.json"
    vibe_json_path = f"dataset/{ds_name}/vibe_cache.json" 
    
    raw_text_dict = {}
    vibe_text_dict = {}
    
    if strategy_name in ["raw", "combination"]:
        if not os.path.exists(raw_json_path):
            raise FileNotFoundError(f"Missing {raw_json_path}. Cannot run TF-IDF.")
        with open(raw_json_path, 'r', encoding='utf-8') as f:
            raw_text_dict = json.load(f)
            
    if strategy_name in ["vibe_only", "combination"]:
        if not os.path.exists(vibe_json_path):
            raise FileNotFoundError(f"Missing {vibe_json_path}! TF-IDF needs the raw LLM text saved as a JSON.")
        with open(vibe_json_path, 'r', encoding='utf-8') as f:
            vibe_text_dict = json.load(f)

    id2token = dataset.field2id_token['item_id']

    # 2. Build Text Corpus
    print("Pre-processing text for TF-IDF...")
    item_texts = []
    valid_count = 0
    
    # Map internal IDs to Amazon String IDs
    id2token = dataset.field2id_token['item_id']
    
    for item_id in range(item_num):
        if item_id == 0:
            item_texts.append("")
            continue
            
        amazon_id = str(id2token[item_id]) # E.g., "B00000J4EY"
        internal_id = str(item_id)          # E.g., "123"
        
        text_parts = []
        
        # Strategy: Use Amazon ID for Raw text, Internal ID for Vibe text
        if strategy_name in ["raw", "combination"]:
            text_parts.append(raw_text_dict.get(amazon_id, ""))
            
        if strategy_name in ["vibe_only", "combination"]:
            text_parts.append(vibe_text_dict.get(internal_id, ""))
            
        final_text = " ".join(text_parts).strip()
        
        if final_text:
            valid_count += 1
            
        item_texts.append(final_text)

    print(f"DEBUG: Found {valid_count} items with non-empty text out of {item_num} total items.")

    # 3. Generate TF-IDF Vectors
    print("Vectorizing corpus using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    all_vectors = vectorizer.fit_transform(item_texts)

    warm_vectors = all_vectors[warm_item_ids]
    cold_vectors = all_vectors[cold_item_ids]

    # 4. Compute Nearest Neighbors efficiently using Scikit-Learn
    print(f"Calculating Top-{k_neighbors} neighbors using sparse matrices...")
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', algorithm='brute')
    nn_model.fit(warm_vectors)
    
    distances, indices = nn_model.kneighbors(cold_vectors)
    
    with torch.no_grad():
        for i, cold_id in enumerate(tqdm(cold_item_ids, desc="Imputing")):
            top_k_indices = indices[i]
            neighbor_actual_ids = warm_item_ids[top_k_indices]
            
            neighbor_weights = model.item_embedding.weight.data[neighbor_actual_ids]
            model.item_embedding.weight.data[cold_id] = torch.mean(neighbor_weights, dim=0)

    print("TF-IDF Imputation complete!")
    
    model.restore_user_e = None
    model.restore_item_e = None
    
    state['model'] = model
    return state

def apply_knn(payload, threshold=5, k_neighbors=5, strategy_name='raw', embed_model='nomic-embed-text'):
    print(f"\n--- Applying FAISS KNN Semantic Imputation (Threshold < {threshold}) ---")
    state = copy.copy(payload)
    model = state['model']
    dataset = state['dataset']
    ds_name = state['config']['dataset']
    
    item_num = dataset.item_num
    train_inter = state['train_data'].dataset.inter_feat
    item_counts = np.bincount(train_inter['item_id'].numpy(), minlength=item_num)
    
    warm_item_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_item_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]
    
    print(f"Dataset stats: {len(warm_item_ids)} Warm items | {len(cold_item_ids)} Cold items.")
    
    if len(cold_item_ids) == 0:
        return state

    cache_file = f"dataset/{ds_name}/{ds_name}_{strategy_name}_embeddings.npy"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Missing {cache_file}. Run build_semantic_caches.py first.")
        
    print(f"Loading cached embeddings from {cache_file}...")
    all_embeddings = np.load(cache_file).astype(np.float32)

    print("Building FAISS index with warm items...")
    warm_embeddings = all_embeddings[warm_item_ids].copy()
    cold_embeddings = all_embeddings[cold_item_ids].copy()
    
    faiss.normalize_L2(warm_embeddings)
    faiss.normalize_L2(cold_embeddings)
    
    embed_dim = warm_embeddings.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    index.add(warm_embeddings)
    
    print(f"Querying top {k_neighbors} neighbors for cold items...")
    distances, indices = index.search(cold_embeddings, k=k_neighbors)
    
    print("Overwriting PyTorch embedding weights...")
    with torch.no_grad():
        for i, cold_id in enumerate(tqdm(cold_item_ids, desc="Imputing")):
            neighbor_actual_ids = warm_item_ids[indices[i]]
            neighbor_weights = model.item_embedding.weight.data[neighbor_actual_ids]
            imputed_weight = torch.mean(neighbor_weights, dim=0)
            model.item_embedding.weight.data[cold_id] = imputed_weight

    print("KNN Semantic Imputation complete!")
    
    # FLUSH THE CACHE
    model.restore_user_e = None
    model.restore_item_e = None
    
    state['model'] = model
    return state


def apply_graph_grafting(payload, threshold=5, strategy_name='vibe_only', embed_model='nomic-embed-text'):
    """
    Direction 2: User History Grafting for LightGCN.
    Finds the 1-NN semantic neighbor, copies its incident user edges to the cold item,
    and recalculates the LightGCN Normalized Laplacian matrix.
    """
    print(f"\n--- Applying Graph Grafting ({strategy_name.upper()}) ---")
    state = copy.copy(payload)
    model = state['model']
    dataset = state['dataset']
    ds_name = state['config']['dataset']
    
    item_num = dataset.item_num
    user_num = dataset.user_num
    train_inter = state['train_data'].dataset.inter_feat
    
    users = train_inter['user_id'].numpy()
    items = train_inter['item_id'].numpy()
    
    item_counts = np.bincount(items, minlength=item_num)
    warm_item_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_item_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]
    
    if len(cold_item_ids) == 0:
        print("No cold items found. Skipping grafting.")
        return state

    cache_file = f"dataset/{ds_name}/{ds_name}_{strategy_name}_embeddings.npy"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Embeddings not found for {strategy_name}. Run build script first.")
        
    all_embeddings = np.load(cache_file).astype(np.float32)
    warm_embeddings = all_embeddings[warm_item_ids].copy()
    cold_embeddings = all_embeddings[cold_item_ids].copy()
    
    faiss.normalize_L2(warm_embeddings)
    faiss.normalize_L2(cold_embeddings)
    
    index = faiss.IndexFlatIP(warm_embeddings.shape[1])
    index.add(warm_embeddings)
    
    print("Querying 1-NN for edge grafting...")
    distances, indices = index.search(cold_embeddings, k=1)

    print("Grafting user histories from warm neighbors to cold items...")
    new_users = []
    new_items = []
    
    for i, cold_id in enumerate(cold_item_ids):
        warm_neighbor_id = warm_item_ids[indices[i][0]]
        neighbor_users = users[items == warm_neighbor_id]
        new_users.extend(neighbor_users)
        new_items.extend([cold_id] * len(neighbor_users))
        
    all_users = np.concatenate([users, new_users])
    all_items = np.concatenate([items, new_items])

    print(f"Rebuilding bipartite graph with {len(new_users)} synthetic edges...")
    R = sp.coo_matrix((np.ones(len(all_users)), (all_users, all_items)), shape=(user_num, item_num), dtype=np.float32)
    
    # ⚡ FAST GRAPH REBUILD: Block matrix construction is instantly evaluated in C++
    A = sp.bmat([[None, R], [R.T, None]], format='coo')
    
    rowsum = np.array(A.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(A).dot(d_mat)
    
    coo = norm_adj.tocoo()
    torch_indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    torch_values = torch.FloatTensor(coo.data)
    shape = coo.shape
    
    sparse_tensor = torch.sparse.FloatTensor(
        torch_indices, torch_values, torch.Size(shape)
    ).coalesce().to(model.device)
    
    # Overwrite LightGCN's internal graph before evaluation (FIXED ATTRIBUTE NAME)
    model.norm_adj_matrix = sparse_tensor
    
    # FLUSH THE CACHE
    model.restore_user_e = None
    model.restore_item_e = None
    
    print("Graph Grafting complete!")
    state['model'] = model
    return state


# =========================================================
# THE PROJECTION NETWORK (For Baseline Comparison)
# =========================================================
class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingMapper, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),         
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),         
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def apply_contrastive_mapper(payload, threshold=5, strategy_name='raw', embed_model='nomic-embed-text', epochs=100):
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

    cache_file = f"dataset/{ds_name}/{ds_name}_{strategy_name}_embeddings.npy"
    all_embeddings = np.load(cache_file)
    
    device = model.device
    X_warm = torch.tensor(all_embeddings[warm_item_ids], dtype=torch.float32).to(device)
    X_cold = torch.tensor(all_embeddings[cold_item_ids], dtype=torch.float32).to(device)
    Y_warm_true = model.item_embedding.weight.data[warm_item_ids].clone().detach().to(device)
    
    X_warm = torch.nn.functional.normalize(X_warm, p=2, dim=1)
    X_cold = torch.nn.functional.normalize(X_cold, p=2, dim=1)

    mapper = EmbeddingMapper(X_warm.shape[1], Y_warm_true.shape[1]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mapper.parameters(), lr=0.001)
    
    batch_size = min(1024, len(warm_item_ids)) 
    train_dataset = data.TensorDataset(X_warm, Y_warm_true)
    dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Learning mapping via InfoNCE (CLIP) Loss...")
    mapper.train()
    temperature = 0.07 
    
    for epoch in tqdm(range(epochs), desc="Training Mapper"):
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            anchor = mapper(batch_X)
            pred_norm = nn.functional.normalize(anchor, p=2, dim=1)
            actual_norm = nn.functional.normalize(batch_Y, p=2, dim=1)
            sim_matrix = torch.matmul(pred_norm, actual_norm.T) / temperature
            labels = torch.arange(len(batch_X)).to(device)
            loss = criterion(sim_matrix, labels)
            loss.backward()
            optimizer.step()
            
    print("Injecting Cold Items via InfoNCE Projection...")
    mapper.eval()
    with torch.no_grad():
        Y_cold_imputed = mapper(X_cold)
        target_magnitude = torch.norm(Y_warm_true, p=2, dim=1).median()
        Y_cold_imputed = torch.nn.functional.normalize(Y_cold_imputed, p=2, dim=1) * target_magnitude
        model.item_embedding.weight.data[cold_item_ids] = Y_cold_imputed
        
    model.restore_user_e = None
    model.restore_item_e = None
        
    state['model'] = model
    return state

