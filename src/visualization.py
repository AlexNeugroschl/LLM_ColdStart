import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_space(state, threshold=5, title="Latent Space PCA", filename="pca_plot.png"):
    """
    Extracts item embeddings from the RecBole state, runs a 2D PCA, and saves a scatter plot 
    highlighting the difference between Warm and Cold items.
    """
    print(f"\n--- Generating PCA Plot: {title} ---")
    
    model = state['model']
    dataset = state['dataset']
    
    # 1. Identify Warm vs Cold Items
    item_num = dataset.item_num
    train_inter = state['train_data'].dataset.inter_feat
    item_counts = np.bincount(train_inter['item_id'].numpy(), minlength=item_num)
    
    # Skip index 0 (RecBole padding)
    warm_ids = np.where((item_counts >= threshold) & (np.arange(item_num) > 0))[0]
    cold_ids = np.where((item_counts < threshold) & (np.arange(item_num) > 0))[0]
    
    # 2. Extract PyTorch Vectors
    # Move to CPU, detach from graph, convert to numpy
    all_vectors = model.item_embedding.weight.data.cpu().numpy()
    
    warm_vectors = all_vectors[warm_ids]
    cold_vectors = all_vectors[cold_ids]
    
    # 3. Fit PCA
    # Best Practice: We fit the PCA ONLY on the Warm items to establish the "True" shape of the latent space.
    # Then we project the Cold items into that space.
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    
    print("Fitting PCA on Warm Items...")
    warm_scaled = scaler.fit_transform(warm_vectors)
    warm_pca = pca.fit_transform(warm_scaled)
    
    print("Projecting Cold Items...")
    cold_scaled = scaler.transform(cold_vectors)
    cold_pca = pca.transform(cold_scaled)
    
    # 4. Generate the Chart
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Plot Warm items (Background: small, gray, low opacity)
    plt.scatter(warm_pca[:, 0], warm_pca[:, 1], 
                alpha=0.3, color='gray', s=20, label=f'Warm Items (n={len(warm_ids)})', edgecolors='none')
    
    # Plot Cold items (Foreground: slightly larger, bright red, high opacity)
    plt.scatter(cold_pca[:, 0], cold_pca[:, 1], 
                alpha=0.8, color='red', s=40, label=f'Cold Items (n={len(cold_ids)})', edgecolors='white', linewidth=0.5)
    
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    plt.legend(loc='upper right', frameon=True)
    
    # Clean up axes limits to focus on the dense cluster (removes extreme outliers)
    plt.xlim(np.percentile(warm_pca[:, 0], [1, 99]))
    plt.ylim(np.percentile(warm_pca[:, 1], [1, 99]))
    
    # 5. Save the file
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Plot saved successfully to {save_path}")

def plot_knn_sweep(k_values, hit_scores, ndcg_scores, title="KNN Hyperparameter Sweep", filename="knn_sweep.png"):
    """
    Generates a dual-axis line chart showing how Hit@10 and NDCG@10 scale with different values of K.
    """
    print(f"\n--- Generating Sweep Plot: {title} ---")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # X-axis
    ax1.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax1.set_xticks(k_values)
    
    # Left Y-Axis (Hit@10)
    color1 = 'tab:blue'
    ax1.set_ylabel('Hit@10', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(k_values, hit_scores, color=color1, marker='o', linewidth=2, markersize=8, label='Hit@10')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Right Y-Axis (NDCG@10)
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('NDCG@10', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(k_values, ndcg_scores, color=color2, marker='s', linewidth=2, markersize=8, linestyle='--', label='NDCG@10')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True)

    plt.title(title, fontsize=16, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the file
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Plot saved successfully to {save_path}")