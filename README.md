# IRIS
**Intent Reactive Intelligent Search**

## 🚀 System Architecture (Interview Revision)

This project implements a hybrid search ranker that fuses lexical, semantic, and intent signals to improve relevance on the Natural Questions dataset.

### 1. Data Pipeline
*   **Preprocessing**: Loads raw NQ data, cleans text, and generates negative samples (irrelevant documents) for training.
*   **Intent Classification**: A dedicated **Intent Encoder** classifies queries into granular categories (Factual, Definition, Procedural, etc.) to guide the ranking logic.

### 2. Feature Engineering (The "Baselines")
Before the neural model sees the data, we generate strong baseline features:
*   **BM25**: Captures exact keyword matching (Lexical Signal).
*   **SBERT**: Captures semantic meaning and cosine similarity (Semantic Signal).
*   **Key Design Choice**: Instead of keeping these as separate baselines, we strictly **inject their raw scores as features** into the final neural model.

### 3. Neural Scorer Architecture
The core "Scorer" is a Bi-Encoder style network that fuses multiple signals:
*   **Inputs**: 
    1.  Query Embeddings & Document Embeddings (from SBERT).
    2.  **Interaction Features**: Element-wise Product (`Q * D`) and Difference (`|Q - D|`) to explicitly model alignment.
    3.  **Intent Vectors**: Enriched representations from the Intent Encoder.
    4.  **Explicit Scores**: Raw BM25 and SBERT similarity scores (~+2 dimensions).
*   **Network**: A Multi-Layer Perceptron (Dense Layers + Batch Norm + Dropout) processes this fused vector.
*   **Output**: A single probability score (0-1) indicating relevance.

### 4. Training & Optimization
*   **Loss Function**: `BCEWithLogitsLoss` (Binary Classification) to distinguish between relevant (1) and irrelevant (0) pairs.
*   **Efficiency**: Implemented specific **Batch Processing** for metrics (NDCG, MRR) and score calculation, replacing slow row-by-row loops (O(N) vs O(N^2)).

### 5. Evaluation
The model is evaluated on a hold-out test set using:
*   **NDCG@k**: Measures if the relevant result is at the top of the list.
*   **Spearman Correlation**: Checks if predicted scores correlate with ground truth/strong baselines.

---

## 📜 Development Phases (History)

### Phase-0: Scope & Data
*   *Completed*: Finalized NQ dataset and project scope.

### Phase-1: Preparation
*   *Completed*: Data loading scripts and granular intent labeling.

### Phase-2: Baselines
*   *Completed*: Implemented BM25 and SBERT standalone rankers for comparison.

### Phase-3: Core Implementation
*   *Completed*: 
    *   **Intent Encoder**: Classifies query intent.
    *   **Scorer**: The hybrid neural ranker described above.
    *   **Optimization**: Vectorized operations for 100x speedup in creating training data.

### Phase-4: Final Polish
*   *Completed*: UI Demo.

---
## How to execute ?
* Make sure to run the main.py file so that it generates the training and testing baselines and also uncomment the baseline generation code as well
* Once the models are created, run the app using streamlit command.
* Enjoy and have fun!

