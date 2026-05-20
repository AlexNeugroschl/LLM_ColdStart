import os
import torch
import numpy as np
import glob
import warnings

# ==========================================
# SILENCE THIRD-PARTY WARNINGS
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Catch the specific Pandas Copy-on-Write warning
warnings.filterwarnings("ignore", message=".*ChainedAssignmentError.*") 
warnings.filterwarnings("ignore", module="recbole.*")

# --- PYTORCH 2.6 HOTFIX FOR RECBOLE ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# --------------------------------------

from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import get_trainer

# Assuming TF-IDF is exported from your modifiers file!
from src.modifiers import apply_knn, apply_contrastive_mapper, apply_graph_grafting, apply_tfidf_knn

class ColdStartExperiment:
    def __init__(self, dataset_name, model_name='LightGCN', is_oracle=False, seed=2024):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.is_oracle = is_oracle
        self.seed = seed
        self.saved_model_path = None 

    def prepare_base_model(self):
        """Trains the base model ONCE per seed and caches the weights to disk."""
        mode_text = "ORACLE (Global Random Mix)" if self.is_oracle else "STRICT HOLDOUT (User Sequential)"
        print(f"\n{'='*70}")
        print(f" TRAINING BASE MODEL: {self.model_name} | {self.dataset_name} | Seed: {self.seed} [{mode_text}] ")
        print(f"{'='*70}")
        
        # Dynamically assign the split architecture
        if self.is_oracle:
            split_args = {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'mode': 'full'}
            config_dict = {
                'seed': self.seed,
                'state': 'WARNING',       
                'show_progress': False,
                'show_progress': True,   
                'eval_args': split_args    
            }
        else:
            split_args = {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'TO', 'mode': 'full'}
            config_dict = {
                'seed': self.seed,
                'state': 'WARNING',       
                'show_progress': True,   
                'eval_args': split_args,
                # Force RecBole to load the timestamp column so it can sort chronologically
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp']} 
            }
        
        # YAML file explicitly attached so LightGCN gets its deep architecture
        run_recbole(
            model=self.model_name, 
            dataset=self.dataset_name, 
            config_file_list=['configs/base_recbole.yaml'], 
            config_dict=config_dict
        )
        
        checkpoints = glob.glob(os.path.join('saved', '*.pth'))
        self.saved_model_path = max(checkpoints, key=os.path.getmtime)
        
        print(f"✅ Base model trained and cached to disk at: {self.saved_model_path}")

    def filter_for_strict_cold_items(self, train_data, test_data, max_train_clicks=5):
        """
        Intercepts the PyTorch test dataset and drops any item that had > max_train_clicks.
        Manually rebuilds RecBole's cache in a version-agnostic way.
        """
        print(f"❄️ Applying Strict Cold Filter (Max {max_train_clicks} train clicks)...")
        
        uid_field = test_data.dataset.uid_field
        iid_field = test_data.dataset.iid_field
        
        # Extract item IDs from RecBole's internal Interaction tensors
        train_items = train_data.dataset.inter_feat[iid_field].cpu().numpy()
        test_items = test_data.dataset.inter_feat[iid_field].cpu().numpy()
        
        original_test_size = len(test_items)
        
        # Identify which items are officially "Warm"
        unique_train_items, counts = np.unique(train_items, return_counts=True)
        warm_items = unique_train_items[counts > max_train_clicks]
        
        # Create a boolean mask: Keep rows where test item is NOT in the warm_items array
        cold_mask = torch.tensor(~np.isin(test_items, warm_items))
        
        # 1. Update the underlying dataset
        test_data.dataset.inter_feat = test_data.dataset.inter_feat[cold_mask]
        
        # 2. MANUALLY REBUILD RECBOLE'S EVALUATION CACHE
        filtered_users = test_data.dataset.inter_feat[uid_field].cpu().numpy()
        filtered_items = test_data.dataset.inter_feat[iid_field].cpu().numpy()
        
        user_num = test_data.dataset.user_num
        uid2items = [[] for _ in range(user_num)]
        
        # Group the filtered items by user
        for u, i in zip(filtered_users, filtered_items):
            uid2items[u].append(i)
            
        # Convert to a NumPy array of PyTorch tensors (dtype=object)
        test_data.uid2positive_item = np.array(
            [torch.tensor(items, dtype=torch.int64) for items in uid2items], 
            dtype=object
        )
        
        # Update the item counts per user
        test_data.uid2items_num = np.array([len(items) for items in uid2items], dtype=np.int64)
        
        new_test_size = len(test_data.dataset.inter_feat)
        print(f"✂️ Filtered Test Set: {original_test_size} -> {new_test_size} cold interactions.")
        
        return test_data

    def run_evaluation(self, modifier_func=None, name="Baseline"):
        if self.saved_model_path is None:
            self.prepare_base_model()

        print(f"\n--- Running Evaluation: {name} ---")
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=self.saved_model_path
        )
        
        current_state = {
            'config': config,
            'model': model,
            'dataset': dataset,
            'train_data': train_data,
            'test_data': test_data
        }
        
        if modifier_func:
            current_state = modifier_func(current_state)
            
        print("Evaluating model on test set...")
        model = current_state['model']
        test_data = current_state['test_data']
        train_data = current_state['train_data'] 
        
        # Force the test set to ONLY evaluate cold items, unless it's the Oracle ceiling
        if not self.is_oracle:
            test_data = self.filter_for_strict_cold_items(train_data, test_data, max_train_clicks=3)
        
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        return test_result

def run_statistical_suite(dataset_name='ml-100k', seeds=None):
    if seeds is None:
        seeds = [42, 101, 777, 2024, 9999]
        
    print(f"\n{'='*80}")
    print(f" STARTING LIGHTGCN MASTER STATISTICAL SUITE: {dataset_name.upper()}")
    print(f" Seeds to evaluate: {seeds}")
    print(f"{'='*80}")

    architectures = [
        "lgcn_oracle",
        "lgcn_baseline",          
        "lgcn_tf_idf",            # NEW: Lexical Baseline
        "lgcn_mapper_net",
        "lgcn_1-nn_raw",
        "lgcn_1-nn_vibe_only",
        "lgcn_1-nn_combination",  # NEW: 1-NN Hybrid
        "lgcn_graft_raw",
        "lgcn_graft_vibe_only",
        "lgcn_graft_combination"
    ]
    
    results = {arch: {'hit': [], 'ndcg': []} for arch in architectures}

    for idx, seed in enumerate(seeds):
        print(f"\n\n{'#'*60}")
        print(f" RUNNING ACADEMIC SUITE {idx+1}/{len(seeds)} (SEED: {seed})")
        print(f"{'#'*60}")
        
        # 1. ORACLE CEILING
        oracle_exp = ColdStartExperiment(dataset_name, model_name='LightGCN', is_oracle=True, seed=seed)
        oracle_exp.prepare_base_model()
        orc_res = oracle_exp.run_evaluation(name="LightGCN Oracle Baseline")
        results["lgcn_oracle"]['hit'].append(orc_res['hit@10'])
        results["lgcn_oracle"]['ndcg'].append(orc_res['ndcg@10'])

        # 2. STANDARD MODELS (Train base LightGCN once)
        std_exp = ColdStartExperiment(dataset_name, model_name='LightGCN', is_oracle=False, seed=seed)
        std_exp.prepare_base_model()
        
        # A. STRICT DO-NOTHING BASELINE
        base_res = std_exp.run_evaluation(modifier_func=None, name="Strict Baseline (No Imputation)")
        results["lgcn_baseline"]['hit'].append(base_res['hit@10'])
        results["lgcn_baseline"]['ndcg'].append(base_res['ndcg@10'])
        
        # B. TF-IDF LEXICAL BASELINE (K=1 for apples-to-apples comparison)
        tfidf_mod = lambda state: apply_tfidf_knn(state, k_neighbors=1)
        tfidf_res = std_exp.run_evaluation(modifier_func=tfidf_mod, name="TF-IDF Lexical Baseline (K=1)")
        results["lgcn_tf_idf"]['hit'].append(tfidf_res['hit@10'])
        results["lgcn_tf_idf"]['ndcg'].append(tfidf_res['ndcg@10'])
        
        # C. Mapper Network (Neural Baseline on Graph)
        map_mod = lambda state: apply_contrastive_mapper(state, strategy_name="raw", epochs=150)
        map_res = std_exp.run_evaluation(modifier_func=map_mod, name="Mapper Network (Raw)")
        results["lgcn_mapper_net"]['hit'].append(map_res['hit@10'])
        results["lgcn_mapper_net"]['ndcg'].append(map_res['ndcg@10'])

        # D. 1-NN Weight Copy (Added Combination)
        for strategy in ["raw", "vibe_only", "combination"]:
            knn_mod = lambda state: apply_knn(state, k_neighbors=1, strategy_name=strategy)
            res = std_exp.run_evaluation(modifier_func=knn_mod, name=f"1-NN Weight Copy ({strategy.upper()})")
            results[f"lgcn_1-nn_{strategy}"]['hit'].append(res['hit@10'])
            results[f"lgcn_1-nn_{strategy}"]['ndcg'].append(res['ndcg@10'])

        # E. Graph Grafting
        for strategy in ["raw", "vibe_only", "combination"]:
            graft_mod = lambda state: apply_graph_grafting(state, strategy_name=strategy)
            res = std_exp.run_evaluation(modifier_func=graft_mod, name=f"Graph Grafting ({strategy.upper()})")
            results[f"lgcn_graft_{strategy}"]['hit'].append(res['hit@10'])
            results[f"lgcn_graft_{strategy}"]['ndcg'].append(res['ndcg@10'])

    # SCORECARD
    def calc_stats(arr):
        return np.mean(arr), np.std(arr)

    print("\n" + "="*125)
    print(f" 🏆 ULTIMATE LIGHTGCN {len(seeds)}-SEED STATISTICAL SCORECARD 🏆")
    print("="*125)
    print(f"{'Model Architecture':<35} | {'Hit@10 (Mean ± SD)':<25} | {'NDCG@10 (Mean ± SD)':<25} | {'Method'}")
    print("-" * 125)
    
    def print_row(key, name_override, note):
        h_m, h_s = calc_stats(results[key]['hit'])
        n_m, n_s = calc_stats(results[key]['ndcg'])
        print(f"{name_override:<35} | {h_m:.4f} ± {h_s:.4f}     | {n_m:.4f} ± {n_s:.4f}     | {note}")

    print_row("lgcn_baseline", "LightGCN Baseline", "No Imputation (Strict Cold)")
    print("-" * 125)
    print_row("lgcn_tf_idf", "LightGCN + TF-IDF", "Lexical Weight Copy (K=1)")
    print_row("lgcn_mapper_net", "LightGCN + Neural Mapper", "Weight Imputation")
    print("-" * 125)
    print_row("lgcn_1-nn_raw", "LightGCN + 1-NN (Raw)", "Weight Copy")
    print_row("lgcn_1-nn_vibe_only", "LightGCN + 1-NN (Vibe Only)", "Weight Copy")
    print_row("lgcn_1-nn_combination", "LightGCN + 1-NN (Hybrid)", "Weight Copy")
    print("-" * 125)
    print_row("lgcn_graft_raw", "LightGCN + Grafting (Raw)", "Edge Injection")
    print_row("lgcn_graft_vibe_only", "LightGCN + Grafting (Vibe Only)", "Edge Injection (Proposed)")
    print_row("lgcn_graft_combination", "LightGCN + Grafting (Hybrid)", "Edge Injection (Proposed)")
    print("-" * 125)
    print_row("lgcn_oracle", "LightGCN Oracle", "Theoretical Maximum")
    print("="*125)



if __name__ == "__main__":
    
    datasets_to_run = [
        # "amazon-office",
        "ml-1m",
        # "steam"
    ]
    
    for dataset in datasets_to_run:
        run_statistical_suite(dataset)