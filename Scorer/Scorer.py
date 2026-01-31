import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from .ScoringModel import ScoringModel
from pathlib import Path
import ast

class Scorer():
    def __init__(self, intentEncoderTrainer, datasetPath: str = None):
        self.intentEncoder = intentEncoderTrainer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataPath = datasetPath
        self.scoringModel = ScoringModel()
    
    def __batchEncode(self, x: pd.DataFrame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        questions = x['question'].tolist()
        long_answers = x['long_answers'].tolist()
        
        # Batch encode with sentence transformer
        query_embeddings = self.encoder.encode(questions, convert_to_tensor=True, batch_size=128, show_progress_bar=False).to(device)
        doc_embeddings = self.encoder.encode(long_answers, convert_to_tensor=True, batch_size=128, show_progress_bar=False).to(device)
        
        # Intent encoder prediction (already batched)
        intent_encoder_output = self.intentEncoder.predict(x)
        
        # Interaction features
        product_embeddings = query_embeddings * doc_embeddings
        diff_embeddings = torch.abs(query_embeddings - doc_embeddings)
        
        # Baseline scores
        # Handle likely missing columns if predicting on raw data without scores (shouldn't happen in this flow but safe to handle)
        if 'bm25_score' not in x.columns:
            bm25_vals = torch.zeros((len(x), 1), device=device)
        else:
            bm25_vals = torch.tensor(x['bm25_score'].fillna(0.0).values, dtype=torch.float32, device=device).unsqueeze(1)
            
        if 'sbert_score' not in x.columns:
            sbert_vals = torch.zeros((len(x), 1), device=device)
        else:
            sbert_vals = torch.tensor(x['sbert_score'].fillna(0.0).values, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Concatenate
        return torch.cat([
            query_embeddings, 
            doc_embeddings,
            intent_encoder_output['query_intent_embed'].to(device), 
            intent_encoder_output['weighted_intent_sum'].to(device), 
            product_embeddings,
            diff_embeddings,
            bm25_vals,
            sbert_vals
        ], dim=1)
    
    def __prepareDataset(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = pd.read_csv(self.dataPath)
        print(f"Preparing dataset with {len(self.data)} rows...")
        feature_embedding_tensor = self.__batchEncode(self.data)
        torch.save(feature_embedding_tensor, 'feature_embedding_tensor_train.pt')
    
    def train(self, epochs: int = 100, learning_rate = 0.001):
        dataset_path = Path('feature_embedding_tensor_train.pt')  # wherever you save the CSV

        if not dataset_path.exists():
            print("Dataset not found. Preparing dataset...")
            self.__prepareDataset()

        t = torch.load('feature_embedding_tensor_train.pt')
        # t is already [N, D], no need to squeeze if saved from batchEncode
        # But wait, original code did: t = t.squeeze(1). 
        # Check if load adds a dimension? No.
        # Original: stack of [1, D] -> [N, 1, D]. So squeeze(1) -> [N, D].
        # New: [N, D]. So NO squeeze needed.
        # However, to be safe against shape mismatch or reloading old files (which I deleted), 
        # I should check shape or just assume [N, D].
        # I'll remove squeeze(1) as my __batchEncode returns [N, D].
        
        print("Dataset loaded.")
        self.data = pd.read_csv('relevance_training_data_with_baselines.csv')
        l= torch.tensor(self.data['relevance_label'].to_list(), dtype=torch.float32).reshape(-1,1)
        
        # Check lengths match
        if len(t) != len(l):
             print(f"Warning: Tensor length {len(t)} != Label length {len(l)}. Re-preparing dataset.")
             self.__prepareDataset()
             t = torch.load('feature_embedding_tensor_train.pt')
        
        loss = nn.BCEWithLogitsLoss()
        optim = torch.optim.Adam(self.scoringModel.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            output = self.scoringModel(t)
            # labels l must be on same device
            if hasattr(t, 'device'):
                l = l.to(t.device)
                
            loss_value = loss(output, l)
            
            optim.zero_grad()
            loss_value.backward(retain_graph=True) # retain_graph needed? intent encoder gradients?
            # Intent encoder is trained separately. Scorer uses its output.
            # Scorer has intentEncoderTrainer.
            # If intentEncoderTrainer graph is connected?
            # predict returns tensors.
            # In __batchEncode: intent_encoder_output = self.intentEncoder.predict(x).
            # If predict returns tensors with grad, then yes.
            # But we are loading from DISK ('feature_embedding_tensor_train.pt').
            # Tensors loaded from disk DO NOT have history/grad_fn unless we manually set requires_grad=True.
            # And we are not optimizing IntentEncoder here, only ScoringModel.
            # So backward() is fine without retain_graph.
            
            optim.step()
            
            print(f"Epoch: {epoch+1} === Loss: {loss_value}")            
        
        # resetting the model to eval mode
        self.scoringModel.eval()
    def predict(self, x: pd.DataFrame):
        print("Predicting relevance scores...")
        # feature_embedding_tensor  = torch.stack(x.apply(self.__encodeText, axis=1).to_list()).squeeze(1)
        feature_embedding_tensor = self.__batchEncode(x)
        return self.scoringModel(feature_embedding_tensor)

    def save(self, model_path: str):
        torch.save(self.scoringModel.state_dict(), model_path)
        print(f"Scorer model saved to {model_path}")

    def load(self, model_path: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ScoringModel is already initialized in __init__, just load weights
        self.scoringModel.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.scoringModel.eval()
        print(f"Scorer model loaded from {model_path}")