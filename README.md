To run the code yourself:
1) Have Ollama installed and pull llama3.2 and nomic-embed-text
2) pip install -r requirements.txt
3) To get datasets and embeddings: python scripts/run_all.py (this will take a while)
4) To train models and run analysis: python main.py
5) To analyze charts: /plot/, tables are printed to terminal







### Table 1: Amazon Office (Strict Cold-Start Evaluation)

| Model Architecture | Hit@10 (Mean ± SD) | Hit@20 (Mean ± SD) | Hit@50 (Mean ± SD) | NDCG@10 (Mean ± SD) | Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGCN Baseline** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | No Imputation (Strict Cold) |
| **LightGCN + TF-IDF (Raw)** | 0.0000 ± 0.0000 | 0.0001 ± 0.0001 | **0.0018 ± 0.0004** | 0.0000 ± 0.0000 | Lexical Copy (K=1) |
| **LightGCN + TF-IDF (LLM Enhanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Lexical Copy (K=1) |
| **LightGCN + TF-IDF (Combo)** | 0.0000 ± 0.0000 | 0.0001 ± 0.0001 | 0.0005 ± 0.0003 | 0.0000 ± 0.0000 | Lexical Copy (K=1) |
| **LightGCN + Neural Mapper** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Imputation |
| **LightGCN + 1-NN (Raw)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Copy |
| **LightGCN + 1-NN (LLM Enhanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Copy |
| **LightGCN + 1-NN (Combo)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Copy |
| **LightGCN + Grafting (Raw)** | 0.0001 ± 0.0001 | 0.0003 ± 0.0003 | 0.0009 ± 0.0002 | 0.0000 ± 0.0000 | Edge Injection |
| **LightGCN + Grafting (LLM Enhanced)** | 0.0000 ± 0.0000 | 0.0001 ± 0.0001 | 0.0012 ± 0.0002 | 0.0000 ± 0.0000 | Edge Injection (Proposed) |
| **LightGCN + Grafting (Combo)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0002 ± 0.0003 | 0.0000 ± 0.0000 | Edge Injection (Proposed) |
| **LightGCN Oracle** | 0.0773 ± 0.0177 | 0.1209 ± 0.0243 | 0.2065 ± 0.0358 | 0.0362 ± 0.0086 | Theoretical Maximum |

<br>

### Table 2: Amazon Digital Music (Strict Cold-Start Evaluation)

| Model Architecture | Hit@10 (Mean ± SD) | Hit@20 (Mean ± SD) | Hit@50 (Mean ± SD) | NDCG@10 (Mean ± SD) | Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGCN Baseline** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | No Imputation (Strict Cold) |
| **LightGCN + TF-IDF (Raw)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Lexical Copy (K=1) |
| **LightGCN + TF-IDF (LLM Enhanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Lexical Copy (K=1) |
| **LightGCN + TF-IDF (Combo)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Lexical Copy (K=1) |
| **LightGCN + Neural Mapper** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Imputation |
| **LightGCN + 1-NN (Raw)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Copy |
| **LightGCN + 1-NN (LLM Enhanced)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Copy |
| **LightGCN + 1-NN (Combo)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Weight Copy |
| **LightGCN + Grafting (Raw)** | 0.0003 ± 0.0000 | 0.0003 ± 0.0000 | 0.0006 ± 0.0000 | 0.0002 ± 0.0000 | Edge Injection |
| **LightGCN + Grafting (LLM Enhanced)** | **0.0019 ± 0.0003** | **0.0032 ± 0.0001** | **0.0043 ± 0.0002** | **0.0003 ± 0.0000** | Edge Injection (Proposed) |
| **LightGCN + Grafting (Combo)** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0014 ± 0.0002 | 0.0000 ± 0.0000 | Edge Injection (Proposed) |
| **LightGCN Oracle** | 0.1843 ± 0.0038 | 0.2718 ± 0.0051 | 0.4200 ± 0.0051 | 0.0882 ± 0.0024 | Theoretical Maximum |