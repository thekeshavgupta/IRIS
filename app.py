import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import plotly.graph_objects as go
import numpy as np

# Import project modules
from DataLoaders.DataLoader import DataLoader
from Baseliner.Baseliner import Baseliner
from Encoder.IntentEncoderTrainer import IntentEncoderTrainer
from Scorer.Scorer import Scorer
from Rankers.BM25Ranker import BM25Ranker

# Page Config
st.set_page_config(page_title="IRIS Search Demo", layout="wide")

@st.cache_resource
def load_and_train_system():
    """
    Loads data and trains the models (IntentEncoder + Scorer) once.
    Returns the ready-to-use Scorer object and the search corpus.
    """
    with st.status("Initializing IRIS System...", expanded=True) as status:
        import os
        
        # 1. Load Search Corpus (Required for inference)
        if os.path.exists('relevance_test_data.csv'):
            st.write("✅ Loading search corpus...")
            test_data = pd.read_csv('relevance_test_data.csv')
            corpus_df = test_data.drop_duplicates(subset=['long_answers']).reset_index(drop=True)
            corpus_texts = corpus_df['long_answers'].tolist()
        else:
            st.error("❌ 'relevance_test_data.csv' not found. Please run main.py first to generate data.")
            st.stop()
            
        # 2. Initialize Models (Inference Only)
        intent_model_path = 'intent_encoder_model.pt'
        intent_classes_path = 'intent_classes.pt'
        scorer_model_path = 'scorer_model.pt'
        
        if os.path.exists(intent_model_path) and os.path.exists(intent_classes_path) and os.path.exists(scorer_model_path):
            st.write("✅ Loading pre-trained models...")
            
            # Initialize without datasetPath
            intent_trainer = IntentEncoderTrainer() 
            intent_trainer.load(intent_model_path, intent_classes_path)
            
            scorer = Scorer(intent_trainer)
            scorer.load(scorer_model_path)
        else:
            st.error("❌ Pre-trained models not found. Please run main.py first to train the system.")
            st.stop()
        
        # 3. Indexing for Retrieval
        st.write("🔍 Indexing for fast retrieval...")
        # BM25 Index
        tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
        bm25_index = BM25Okapi(tokenized_corpus)
        
        # SBert Index
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus_embeddings = sbert_model.encode(corpus_texts, convert_to_tensor=True)
        
        status.update(label="System Ready!", state="complete", expanded=False)
        return scorer, corpus_df, bm25_index, sbert_model, corpus_embeddings

# Load System
try:
    scorer, corpus_df, bm25_index, sbert_model, corpus_embeddings = load_and_train_system()
    st.success("System Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- UI Layout ---
st.title("🌈 IRIS: Intent Reactive Intelligent Search")
st.markdown("""
This dashboard demonstrates the neural re-ranking capabilities of IRIS.
Enter a query to see how the model combines **Lexical** (BM25), **Semantic** (SBERT), and **Intent** signals.
""")

query = st.text_input("Enter your search query:", "who is the active chairman of reliance industries")

if query:
    # 1. First Stage Retrieval (Candidate Generation using BM25)
    # real-world: retrieve top-100. Demo: top-20
    tokenized_query = query.split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Get top 20 candidates indices
    top_n = 20
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]
    
    candidates = corpus_df.iloc[top_indices].copy()
    candidates['bm25_score'] = [bm25_scores[i] for i in top_indices]
    
    # 2. Feature Calculation for Candidates
    # Calculate SBert Scores
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    candidate_embeddings = corpus_embeddings[top_indices]
    
    # Cosine Similarity
    import torch.nn.functional as F
    # normalize
    query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
    cand_norm = F.normalize(candidate_embeddings, p=2, dim=1)
    sbert_scores = torch.mm(query_norm, cand_norm.transpose(0, 1)).squeeze(0).cpu().numpy()
    
    candidates['sbert_score'] = sbert_scores
    
    # Add Query and Intent (placeholder for prediction)
    candidates['question'] = query
    # Intent prediction happens inside scorer.predict -> batchEncode -> intentEncoder.predict
    
    # 3. Neural Re-Ranking
    with st.spinner("Neural Re-Ranking..."):
        # We need to construct the DataFrame as expected by Scorer
        # Scorer expects: question, long_answers, bm25_score, sbert_score
        # It calls batchEncode.
        
        # Note: Scorer.predict expects a DataFrame.
        # Warning: Scorer internal handles intent prediction.
        
        # Use raw logits for ranking
        relevance_logits = scorer.predict(candidates)
        logits = relevance_logits.detach().cpu().numpy().flatten()
        candidates['neural_score'] = logits
        
        # Calculate Relative Probability using Softmax (avoids 1.0 saturation)
        # We use numpy for stable softmax
        exp_logits = np.exp(logits - np.max(logits)) # Shift for stability
        candidates['relevance_prob'] = exp_logits / exp_logits.sum()
    
    # 4. Display Results
    # Sort by Neural Score
    results = candidates.sort_values(by='neural_score', ascending=False).reset_index(drop=True)
    
    # Detect Intent (Get it explicitly for display)
    intent_out = scorer.intentEncoder.predict(pd.DataFrame({'question': [query]}))
    # logits to prob
    intent_probs = F.softmax(intent_out['intent_logits'], dim=1)
    top_intent_idx = torch.argmax(intent_probs).item()
    top_intent = scorer.intentEncoder.intent_classes[top_intent_idx]
    confidence = intent_probs[0][top_intent_idx].item()
    
    st.info(f"🧠 **Detected Intent:** {top_intent} ({confidence:.1%})")
    
    st.subheader(f"Top {top_n} Re-Ranked Results")
    st.caption("ℹ️ **Scores**: Showing 'Relative Probability' (Softmax). This represents how likely this result is the best answer *compared to the others shown*.")
    
    for i, row in results.iterrows():
        with st.expander(f"#{i+1}: {row['long_answers'][:100]}... (Prob: {row['relevance_prob']:.2%})"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Full Text:**")
                st.write(row['long_answers'])
            with col2:
                st.write("**Score Breakdown:**")
                st.metric("Probability", f"{row['relevance_prob']:.2%}")
                st.caption(f"(Raw Logit: {row['neural_score']:.2f})")
                st.write("---")
                st.write(f"**BM25:** {row['bm25_score']:.2f}")
                st.write(f"**SBERT:** {row['sbert_score']:.2f}")
                
            # Visualization bar
            # Normalize scores for visual comparison
            # Simple bar chart
            chart_data = pd.DataFrame({
                'Signal': ['Lexical (BM25)', 'Semantic (SBert)', 'Final (Logit)'],
                'Value': [row['bm25_score'], row['sbert_score'], row['neural_score']]
            })
            st.bar_chart(chart_data, x='Signal', y='Value')
