import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from .ScoringModel import ScoringModel
from pathlib import Path
import ast

class Scorer():
    def __init__(self, intentEncoderTrainer, datasetPath: str):
        self.intentEncoder = intentEncoderTrainer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataPath = datasetPath
        self.scoringModel = ScoringModel()
    
    def __encodeText(self, x: pd.DataFrame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        query_embeddings = self.encoder.encode(x['question'], convert_to_tensor=True).to(device)
        doc_embeddings = self.encoder.encode(x['long_answers'], convert_to_tensor=True).to(device)
        intent_encoder_output = self.intentEncoder.predict(x)
        
        return torch.cat([query_embeddings.unsqueeze(0), intent_encoder_output['query_intent_embed'].to(device), intent_encoder_output['weighted_intent_sum'].to(device), doc_embeddings.unsqueeze(0)], dim=1).to(device)
    
    def __prepareDataset(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = pd.read_csv(self.dataPath)
        feature_embedding_tensor  = torch.stack(self.data.apply(self.__encodeText, axis=1).to_list())
        torch.save(feature_embedding_tensor, 'feature_embedding_tensor_train.pt')
    
    def train(self, epochs: int = 100, learning_rate = 0.001):
        dataset_path = Path('feature_embedding_tensor_train.pt')  # wherever you save the CSV

        if not dataset_path.exists():
            print("Dataset not found. Preparing dataset...")
            self.__prepareDataset()
        else:
            t = torch.load('feature_embedding_tensor_train.pt')
            t = t.squeeze(1)
            print("Dataset already exists. Skipping preparation.")
            self.data = pd.read_csv('relevance_training_data_with_baselines.csv')
            l= torch.tensor(self.data['relevance_label'].to_list(), dtype=torch.float32).reshape(-1,1)
            loss = nn.BCEWithLogitsLoss()
            optim = torch.optim.Adam(self.scoringModel.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                output = self.scoringModel(t)
                loss_value = loss(output, l)
                
                optim.zero_grad()
                loss_value.backward()
                optim.step()
                
                print(f"Epoch: {epoch+1} === Loss: {loss_value}")            
            
            # resetting the model to eval mode
            self.scoringModel.eval()
    
    def predict(self, x: pd.DataFrame):
        feature_encodings = x.apply(self.__encodeText)
        return self.scoringModel(feature_encodings)